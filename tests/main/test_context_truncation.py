"""
Tests for the smart context-overflow truncation in ChatAgent._prep_llm_messages.

Issue #838: Previously every overflowing message was blindly truncated to
a fixed 30-token minimum.  The fix computes the actual number of tokens each
message needs to be reduced by so that the history fits within the context
window, preserving as much content as possible.

The expected per-message retention is checked against an independent,
test-side simulation of the documented contract
(see `docs/notes/context-overflow.md`): walking forward from the oldest
eligible message, the current excess is divided evenly over the remaining
*compressible* messages (those whose content sits above the 30-token floor
plus the warning-marker overhead), each message additionally absorbing
whatever later messages cannot, with the excess recomputed at every step.
"""

import math

import pytest

from langroid.agent.chat_agent import (
    CHAT_HISTORY_BUFFER,
    ChatAgent,
    ChatAgentConfig,
)
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.mock_lm import MockLMConfig
from langroid.parsing.file_attachment import FileAttachment
from langroid.parsing.parser import Parser, ParsingConfig, Splitter
from langroid.utils.configuration import Settings, set_global

# Tests must account for the history buffer:
#   budget = context_length - min_output - CHAT_HISTORY_BUFFER
_MIN_OUTPUT = 64  # explicitly passed to the mock LLM config below

# Must match the values used in ChatAgent._prep_llm_messages
_FLOOR = 30
_WARNING = "\n... [Contents truncated!]"


def _make_agent(
    context_length: int,
    max_output_tokens: int = 4096,
    min_output_tokens: int = _MIN_OUTPUT,
    splitter: str = Splitter.MARKDOWN,
) -> ChatAgent:
    """Return a ChatAgent backed by MockLM with a bounded context window."""
    cfg = ChatAgentConfig(
        name="TruncAgent",
        llm=MockLMConfig(
            chat_context_length=context_length,
            max_output_tokens=max_output_tokens,
            min_output_tokens=min_output_tokens,
            default_response="ok",
        ),
        context_overflow_strategy="truncate",
    )
    agent = ChatAgent(cfg)
    agent.parser = Parser(ParsingConfig(splitter=splitter))
    return agent


def _fill_history(agent: ChatAgent, n_msgs: int, words_each: int) -> None:
    """Populate agent.message_history with alternating user/assistant messages."""
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    ]
    for i in range(n_msgs):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        content = ("word " * words_each).strip()
        agent.message_history.append(LLMMessage(role=role, content=content))


def _budget(context_length: int, min_output: int = _MIN_OUTPUT) -> int:
    return context_length - min_output - CHAT_HISTORY_BUFFER


def _hist_at_truncation(agent: ChatAgent, final_msg: str) -> list[LLMMessage]:
    """The message list `_prep_llm_messages(final_msg)` sees at truncation time.

    `_prep_llm_messages` replaces the system message with a freshly built one
    and appends the final user message before the truncation loop runs;
    mirror that here so expected token counts can be computed up-front.
    """
    return (
        [agent._create_system_and_tools_message()]
        + agent.message_history[1:]
        + [LLMMessage(role=Role.USER, content=final_msg)]
    )


def _simulate_even_share_truncation(
    agent: ChatAgent,
    hist: list[LLMMessage],
    budget: int,
) -> dict[int, int]:
    """Independent (test-side) simulation of the truncation contract.

    Walks the compressible range (index 1 .. last-user-index - 1) as
    documented in `docs/notes/context-overflow.md`: at each step the current
    excess — computed over FULL message token counts, i.e. including any
    attachment payloads — is divided evenly over the remaining messages whose
    *content* can still shrink (content tokens above the 30-token floor plus
    the appended-warning overhead), and the current message additionally
    absorbs whatever later messages cannot.

    Returns:
        Mapping {msg_idx: expected content tokens after truncation} for every
        message the contract says must be truncated.
    """
    parser = agent.parser
    assert parser is not None
    warning_tokens = parser.num_tokens(_WARNING)
    last_user_idx = max(i for i, m in enumerate(hist) if m.role == Role.USER)
    content_tokens = [parser.num_tokens(m.content) for m in hist]
    total = agent.chat_num_tokens(hist)
    expected: dict[int, int] = {}
    for idx in range(1, last_user_idx):
        excess = total - budget
        if excess <= 0:
            break
        if content_tokens[idx] <= _FLOOR + warning_tokens:
            continue  # cannot shrink; the implementation skips it too
        later_caps = [
            content_tokens[j] - _FLOOR - warning_tokens
            for j in range(idx + 1, last_user_idx)
        ]
        n_compressible = 1 + sum(1 for c in later_caps if c > 0)
        share = math.ceil(excess / n_compressible)
        later_capacity = sum(c for c in later_caps if c > 0)
        reduction = max(share, excess - later_capacity)
        keep = max(_FLOOR, content_tokens[idx] - reduction - warning_tokens)
        final = keep + warning_tokens
        total -= content_tokens[idx] - final
        content_tokens[idx] = final
        expected[idx] = final
    return expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_truncation_fits_within_context(test_settings: Settings) -> None:
    """After truncation the history must fit within the context budget."""
    set_global(test_settings)
    # Use a context window large enough for the buffer but small enough to need
    # truncation: budget = 1500 - 64 - 300 = 1136 tokens.
    context_length = 1500
    # 6 messages × 300 words ≈ 1800 tokens → well over budget
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=6, words_each=300)

    total_before = agent.chat_num_tokens()
    assert total_before > _budget(context_length), (
        f"precondition: history ({total_before}) must exceed budget "
        f"({_budget(context_length)})"
    )

    hist, _ = agent._prep_llm_messages(truncate=True)

    assert hist, "expected a non-empty message list after truncation"
    total_after = agent.chat_num_tokens(hist)
    budget = _budget(context_length)
    assert (
        total_after <= budget
    ), f"History still too long: {total_after} tokens (budget={budget})"


def test_smart_truncation_keeps_more_than_30_tokens(test_settings: Settings) -> None:
    """
    When the total overflow is small, smart truncation should leave each
    message with well more than 30 tokens rather than collapsing it.
    """
    set_global(test_settings)
    # budget = 2000 - 64 - 300 = 1636 tokens.
    # 4 messages × 200 words ≈ 800 tokens + system ≈ 810 total.
    # Forcing max_output_tokens = context so reduced output < min_output:
    #   reduced_output = 2000 - 810 - 300 = 890 > min_output → output-only reduction.
    # To reach the message-truncation path we need total + min_output > context.
    # With context=900: budget = 900 - 64 - 300 = 536.
    # 4 msgs × 200 words ≈ 800 → need to remove ~264 tokens across 3 compressible msgs
    # → reduction ≈ 88 per msg; keep ≈ 112 tokens each — well above 30.
    context_length = 900
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=4, words_each=200)

    total_before = agent.chat_num_tokens()
    assert total_before > _budget(context_length), (
        f"precondition: history ({total_before}) must exceed budget "
        f"({_budget(context_length)})"
    )

    hist, _ = agent._prep_llm_messages(truncate=True)

    assert hist, "expected non-empty message list after truncation"
    # Check that the first compressible message (index 1) retained > 30 tokens.
    msg_tokens = agent.parser.num_tokens(hist[1].content)
    assert msg_tokens > 30, (
        f"Smart truncation should retain more than 30 tokens per message "
        f"when overflow is moderate; got {msg_tokens}"
    )


def test_truncation_with_tiny_interleaved_messages(test_settings: Settings) -> None:
    """
    Messages at/below the keep-floor cannot absorb any of the excess
    (truncating them would only *grow* them by the appended warning), so they
    must be skipped AND excluded from the even-share denominator: the two
    large messages must shed approximately EQUAL amounts, rather than the
    first shedding only excess/4 (denominator diluted by the two tiny
    messages) and the second absorbing the rest. The compressible range also
    *ends* on a tiny message, so this doubles as a regression test that the
    loop skips past it without running out of messages and raising ValueError.
    """
    set_global(test_settings)
    context_length = 900
    agent = _make_agent(context_length, max_output_tokens=context_length)
    assert agent.parser is not None
    large = ("word " * 400).strip()
    # After the final user message is appended by _prep_llm_messages below:
    # [SYSTEM, USER(large), ASSISTANT(tiny), USER(large), ASSISTANT(tiny),
    #  USER(final)]: compressible range is 1..4, and it *ends* on a tiny msg.
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=Role.USER, content=large),
        LLMMessage(role=Role.ASSISTANT, content="ok"),
        LLMMessage(role=Role.USER, content=large),
        LLMMessage(role=Role.ASSISTANT, content="ok"),
    ]
    large_tokens = agent.parser.num_tokens(large)

    total_before = agent.chat_num_tokens()
    assert total_before > _budget(context_length), (
        f"precondition: history ({total_before}) must exceed budget "
        f"({_budget(context_length)})"
    )

    hist, _ = agent._prep_llm_messages(message="Final question", truncate=True)

    assert agent.chat_num_tokens(hist) <= _budget(context_length)
    # Tiny messages must be left intact (no truncation warning appended).
    assert hist[2].content == "ok"
    assert hist[4].content == "ok"
    # The final user message must be untouched.
    assert hist[-1].content == "Final question"
    # Both large messages were truncated, ...
    shed_1 = large_tokens - agent.parser.num_tokens(hist[1].content)
    shed_3 = large_tokens - agent.parser.num_tokens(hist[3].content)
    assert shed_1 > 0, "first large message should have been truncated"
    assert shed_3 > 0, "second large message should have been truncated"
    # ... neither was collapsed toward the 30-token floor, ...
    assert agent.parser.num_tokens(hist[1].content) > 100
    assert agent.parser.num_tokens(hist[3].content) > 100
    # ... and they shed approximately EQUAL amounts: the tiny messages must
    # not dilute the even-share denominator. (With the diluted denominator
    # the first message sheds ~excess/4 and the second ~3x that, a gap far
    # beyond this tolerance.)
    assert abs(shed_1 - shed_3) <= 25, (
        f"Excess not evenly distributed across the two compressible "
        f"messages: sheds are {shed_1} vs {shed_3} tokens"
    )


def test_truncation_skips_none_content(test_settings: Settings) -> None:
    """Call-only turns with no content remain intact during truncation."""
    set_global(test_settings)
    context_length = 900
    agent = _make_agent(context_length, max_output_tokens=context_length)
    large = ("word " * 400).strip()
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=Role.USER, content=large),
        LLMMessage(role=Role.ASSISTANT, content=None),
        LLMMessage(role=Role.USER, content=large),
    ]

    hist, _ = agent._prep_llm_messages(message="Final question", truncate=True)

    assert agent.chat_num_tokens(hist) <= _budget(context_length)
    assert hist[2].content is None


def test_truncation_preserves_system_and_last_message(test_settings: Settings) -> None:
    """The system message and the last user message must never be modified."""
    set_global(test_settings)
    context_length = 1500
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=6, words_each=300)

    system_content = agent.message_history[0].content
    # Pass a distinct final user message (with message=None the last history
    # message would be dropped for re-generation, weakening this check).
    final_user_msg = "FINAL distinct user question"

    hist, _ = agent._prep_llm_messages(message=final_user_msg, truncate=True)

    assert hist, "expected non-empty message list"
    assert (
        hist[0].content == system_content
    ), "System message (index 0) must not be truncated"
    assert (
        hist[-1].content == final_user_msg
    ), "Last (user) message must not be truncated"
    # In-between messages were actually truncated.
    assert any("Contents truncated" in m.content for m in hist[1:-1])


def test_truncation_raises_when_impossible(test_settings: Settings) -> None:
    """ValueError must be raised when the context cannot be made to fit."""
    set_global(test_settings)
    # budget = 500 - 490 - 300 = -290 (impossible).
    # The loop will exhaust all compressible messages and raise.
    agent = _make_agent(
        context_length=500, max_output_tokens=500, min_output_tokens=490
    )
    _fill_history(agent, n_msgs=4, words_each=50)

    with pytest.raises(ValueError, match="(?s)run out.*of msgs"):
        agent._prep_llm_messages(truncate=True)


@pytest.mark.parametrize(
    "words_a, words_b, context_length",
    [
        # Equal-size messages: the excess is split evenly between them.
        (400, 400, 900),
        # Second message is small (but above the skip threshold): its
        # capacity caps what it can absorb, so the first message must absorb
        # the remainder beyond its even share (capacity check).
        (400, 60, 700),
    ],
)
def test_prep_llm_messages_even_share_distribution(
    test_settings: Settings,
    words_a: int,
    words_b: int,
    context_length: int,
) -> None:
    """
    End-to-end check of the truncation distribution in `_prep_llm_messages`:
    each compressible message must retain the number of content tokens
    predicted by the independent simulation of the documented even-share +
    capacity contract, within a small tokenizer tolerance.
    """
    set_global(test_settings)
    agent = _make_agent(context_length, max_output_tokens=context_length)
    assert agent.parser is not None
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=Role.USER, content=("word " * words_a).strip()),
        LLMMessage(role=Role.ASSISTANT, content=("word " * words_b).strip()),
    ]
    final_msg = "Final question"
    budget = _budget(context_length)

    sim_hist = _hist_at_truncation(agent, final_msg)
    assert agent.chat_num_tokens(sim_hist) > budget, "precondition: overflow"
    expected = _simulate_even_share_truncation(agent, sim_hist, budget)
    assert set(expected) == {1, 2}, "precondition: both messages get truncated"

    hist, _ = agent._prep_llm_messages(message=final_msg, truncate=True)

    assert agent.chat_num_tokens(hist) <= budget
    for idx, expected_tokens in expected.items():
        assert "Contents truncated" in hist[idx].content
        actual = agent.parser.num_tokens(hist[idx].content)
        assert abs(actual - expected_tokens) <= 12, (
            f"message {idx}: retained {actual} content tokens, expected "
            f"~{expected_tokens} per the even-share/capacity contract"
        )


def test_truncation_attachment_tokens_count_toward_excess(
    test_settings: Settings,
) -> None:
    """
    Attachment payloads must count toward the *overall* excess (which uses
    `chat_num_tokens`), while each message's keep-target must be computed
    from its content-only token count: `truncate_message` can only trim
    content, so a keep-target inflated by attachment tokens would exceed the
    message's content size and leave the attachment-bearing message
    effectively untruncated (over-kept), silently shifting its share of the
    reduction onto later messages.
    """
    set_global(test_settings)
    context_length = 900
    # Use the tiktoken-based splitter so the serialized attachment payload
    # (a base64 blob without whitespace) has a realistic token count; the
    # default MARKDOWN splitter counts whitespace-separated words, under
    # which the whole payload would count as a single token.
    agent = _make_agent(
        context_length,
        max_output_tokens=context_length,
        splitter=Splitter.TOKENS,
    )
    assert agent.parser is not None
    attachment = FileAttachment.from_bytes(
        content=b"fake pdf bytes " * 20,
        filename="report.pdf",
    )
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content=("word " * 200).strip(),
            files=[attachment],
        ),
        LLMMessage(role=Role.ASSISTANT, content=("word " * 220).strip()),
    ]
    final_msg = "What does the attached report say?"
    budget = _budget(context_length)

    sim_hist = _hist_at_truncation(agent, final_msg)
    content_only_total = sum(agent.parser.num_tokens(m.content) for m in sim_hist)
    attachment_tokens = agent.chat_num_tokens(sim_hist) - content_only_total
    assert attachment_tokens > 100, "precondition: sizeable attachment payload"
    # The overflow must be attributable to the attachment payload: the text
    # content alone fits the budget; content + attachment does not.
    assert content_only_total <= budget, "precondition: content alone fits"
    assert (
        agent.chat_num_tokens(sim_hist) > budget
    ), "precondition: attachment payload pushes the history over budget"
    expected = _simulate_even_share_truncation(agent, sim_hist, budget)
    assert 1 in expected, "precondition: attachment-bearing msg gets truncated"

    orig_content_tokens = agent.parser.num_tokens(agent.message_history[1].content)

    hist, _ = agent._prep_llm_messages(message=final_msg, truncate=True)

    # The request fits, counting attachment payload tokens.
    assert agent.chat_num_tokens(hist) <= budget
    # The attachment itself is preserved (only content is trimmed).
    assert len(hist[1].files) == 1
    # The attachment-bearing message's *content* was truncated to its
    # content-only keep-target. Had the keep-target been computed from
    # content + attachment tokens (e.g. via `_message_num_tokens`), it would
    # exceed the content size and the content would be over-kept in full.
    assert "Contents truncated" in hist[1].content
    actual = agent.parser.num_tokens(hist[1].content)
    assert actual < orig_content_tokens
    assert abs(actual - expected[1]) <= 12, (
        f"attachment-bearing message retained {actual} content tokens, "
        f"expected ~{expected[1]}: its keep-target must be based on "
        f"content-only tokens, with the attachment payload still counting "
        f"toward the overall excess"
    )
