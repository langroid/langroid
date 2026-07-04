"""
Tests for the smart context-overflow truncation in ChatAgent._prep_llm_messages.

Issue #838: Previously every overflowing message was blindly truncated to
a fixed 30-token minimum.  The fix computes the actual number of tokens each
message needs to be reduced by so that the history fits within the context
window, preserving as much content as possible.
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
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.utils.configuration import Settings, set_global

# Tests must account for the history buffer:
#   budget = context_length - min_output - CHAT_HISTORY_BUFFER
_MIN_OUTPUT = 64  # explicitly passed to the mock LLM config below


def _make_agent(
    context_length: int,
    max_output_tokens: int = 4096,
    min_output_tokens: int = _MIN_OUTPUT,
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
    agent.parser = Parser(ParsingConfig())
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
    Messages at/below the keep-floor cannot absorb their share of the excess
    (truncating them would only *grow* them by the appended warning). The
    large messages must absorb the leftover instead — even when the last
    compressible message is tiny — rather than the loop running out of
    messages and raising ValueError (regression test).
    """
    set_global(test_settings)
    context_length = 900
    agent = _make_agent(context_length, max_output_tokens=context_length)
    # [SYSTEM, USER(large), ASSISTANT(tiny), USER(large), ASSISTANT(tiny),
    #  USER(final)]: compressible range is 1..4, and it *ends* on a tiny msg.
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=Role.USER, content=("word " * 400).strip()),
        LLMMessage(role=Role.ASSISTANT, content="ok"),
        LLMMessage(role=Role.USER, content=("word " * 400).strip()),
        LLMMessage(role=Role.ASSISTANT, content="ok"),
        LLMMessage(role=Role.USER, content="Final question"),
    ]

    total_before = agent.chat_num_tokens()
    assert total_before > _budget(context_length), (
        f"precondition: history ({total_before}) must exceed budget "
        f"({_budget(context_length)})"
    )

    hist, _ = agent._prep_llm_messages(truncate=True)

    assert agent.chat_num_tokens(hist) <= _budget(context_length)
    # Tiny messages must be left intact (no truncation warning appended).
    assert hist[2].content == "ok"
    assert hist[4].content == "ok"
    # Large messages absorbed the excess but kept well above the 30 floor.
    assert agent.parser is not None
    assert agent.parser.num_tokens(hist[1].content) > 30


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


def test_truncate_message_smart_target(test_settings: Settings) -> None:
    """
    Unit-test the per-message token target computation: given an excess and a
    number of remaining compressible messages, each message should be reduced
    by ceil(excess / n_remaining) tokens (but not below 30).
    """
    set_global(test_settings)
    agent = _make_agent(context_length=10_000)
    _fill_history(agent, n_msgs=4, words_each=200)

    msg_idx = 1
    original_tokens = agent._message_num_tokens(agent.message_history[msg_idx])

    # Simulate: 300 tokens of excess to spread across 3 remaining messages.
    excess = 300
    n_remaining = 3
    reduction = math.ceil(excess / n_remaining)  # 100
    keep = max(30, original_tokens - reduction)

    truncated = agent.truncate_message(msg_idx, tokens=keep, inplace=False)
    actual_tokens = agent.parser.num_tokens(truncated.content)

    assert (
        actual_tokens <= keep + 5
    ), (  # small tolerance for tiktoken approximation
        f"Truncated message has {actual_tokens} tokens, expected ≤ {keep + 5}"
    )
    assert actual_tokens >= 20, "Truncated message should not be empty"
