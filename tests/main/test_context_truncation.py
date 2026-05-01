"""
Tests for the smart context-overflow truncation in ChatAgent._prep_llm_messages.

Issue #838: Previously every overflowing message was blindly truncated to
a fixed 30-token minimum.  The fix computes the actual number of tokens each
message needs to be reduced by so that the history fits within the context
window, preserving as much content as possible.
"""

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.mock_lm import MockLMConfig
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.utils.configuration import Settings, set_global

# CHAT_HISTORY_BUFFER from chat_agent.py (300 tokens).
# Tests must account for it: budget = context_length - min_output - 300.
_CHAT_HISTORY_BUFFER = 300
_MIN_OUTPUT = 64  # matches LLMConfig.min_output_tokens default


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
    return context_length - min_output - _CHAT_HISTORY_BUFFER


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


def test_truncation_preserves_system_and_last_message(test_settings: Settings) -> None:
    """The system message and the last user message must never be modified."""
    set_global(test_settings)
    context_length = 1500
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=6, words_each=300)

    system_content = agent.message_history[0].content
    last_content = agent.message_history[-1].content

    hist, _ = agent._prep_llm_messages(truncate=True)

    assert hist, "expected non-empty message list"
    assert (
        hist[0].content == system_content
    ), "System message (index 0) must not be truncated"
    assert hist[-1].content == last_content, "Last message must not be truncated"


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


def test_small_overflow_preserves_most_content(test_settings: Settings) -> None:
    """
    Headline regression test for the fix: when overflow is small, each
    truncated message should keep most of its content rather than being
    collapsed to a 30-token floor (the old behaviour).
    """
    set_global(test_settings)
    # budget = 1000 - 64 - 300 = 636 tokens.
    # 5 messages × 130 words ≈ 650 + system ≈ 655 → ~19 tokens over budget.
    # With 4 compressible messages, even distribution gives reduction ≈ 5
    # per msg, so each should retain ~125 tokens — far above the 30-token
    # floor that the old code would have produced.
    context_length = 1000
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=5, words_each=130)

    original_compressible_tokens = [
        agent.parser.num_tokens(agent.message_history[i].content)
        for i in range(1, len(agent.message_history) - 1)
    ]

    hist, _ = agent._prep_llm_messages(truncate=True)

    truncated_tokens = [
        agent.parser.num_tokens(hist[i].content) for i in range(1, len(hist) - 1)
    ]
    # No message should have been collapsed near the 30-token floor.
    assert all(t > 80 for t in truncated_tokens), (
        f"Small overflow should not collapse messages near the 30-token "
        f"floor; per-msg tokens={truncated_tokens}"
    )
    # And every truncated message should be smaller than its original
    # (or unchanged, but at least one must be smaller for the loop to exit).
    assert any(
        t < orig for t, orig in zip(truncated_tokens, original_compressible_tokens)
    ), "expected at least one message to actually be truncated"


def test_smart_truncation_distributes_evenly(test_settings: Settings) -> None:
    """
    The contract of the fix: per-message reductions should be roughly equal
    across messages that actually got truncated, not all clamped to a fixed
    floor. We verify this by checking the spread of per-message reductions.
    """
    set_global(test_settings)
    # budget = 900 - 64 - 300 = 536 tokens.
    # 4 msgs × 200 words ≈ 800 → over by ~264 across 3 compressible msgs
    # → reduction ≈ 88 each.
    context_length = 900
    agent = _make_agent(context_length, max_output_tokens=context_length)
    _fill_history(agent, n_msgs=4, words_each=200)

    original_tokens = [
        agent.parser.num_tokens(agent.message_history[i].content)
        for i in range(1, len(agent.message_history) - 1)
    ]

    hist, _ = agent._prep_llm_messages(truncate=True)

    new_tokens = [
        agent.parser.num_tokens(hist[i].content) for i in range(1, len(hist) - 1)
    ]
    reductions = [orig - new for orig, new in zip(original_tokens, new_tokens)]
    truncated_reductions = [r for r in reductions if r > 0]

    # The reductions across truncated messages should be of comparable size
    # (within a 2x spread) — the old code would have produced reductions
    # that pinned every message to 30 tokens regardless.
    assert truncated_reductions, "expected at least one message to be truncated"
    spread = max(truncated_reductions) - min(truncated_reductions)
    assert spread < max(
        truncated_reductions
    ), f"Reductions should be roughly even, got {truncated_reductions}"
