import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import OpenAIGPTConfig

CHAT_CONTEXT_LENGTH = 16_000
MAX_OUTPUT_TOKENS = 1000
MIN_OUTPUT_TOKENS = 50


@pytest.fixture
def agent():
    """Create a ChatAgent with a mock LLM for testing truncation."""
    config = ChatAgentConfig(
        system_message="System message",
        llm=OpenAIGPTConfig(
            # Small context for testing truncation
            chat_context_length=CHAT_CONTEXT_LENGTH,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            min_output_tokens=MIN_OUTPUT_TOKENS,
        ),
    )
    agent = ChatAgent(config)

    # Create a mock parser that counts tokens as characters for simplicity
    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            else:
                return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    # Create a mock LLM that returns a fixed context length
    class MockLLM:
        def chat_context_length(self):
            return CHAT_CONTEXT_LENGTH

        def supports_functions_or_tools(self):
            return False

    agent.llm = MockLLM()

    # Initialize message history with a system message
    # agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]
    agent.init_message_history()
    return agent


def test_no_truncation_needed(agent):
    """Test when no truncation is needed."""
    # Add a short user message (well within context limits)
    message = "Short user message"

    # Call the method
    hist, output_len = agent._prep_llm_messages(message)

    # History should include system message and the new user message
    assert len(hist) == 2
    assert hist[0].content == "System message"
    assert hist[1].content == message
    assert output_len == MAX_OUTPUT_TOKENS  # Original max output tokens


def test_reduce_output_length(agent):
    """Test when only output length reduction is needed."""
    # Fill most of the context with long messages
    long_message = "X" * 15_000  # 700 tokens
    agent.message_history.append(LLMMessage(role=Role.USER, content=long_message))

    # New user message
    message = "Another message"

    # Call the method
    hist, output_len = agent._prep_llm_messages(message)

    # Check that output length was reduced but no messages were truncated
    assert len(hist) == 3
    assert hist[1].content == long_message  # Not truncated
    assert output_len < MAX_OUTPUT_TOKENS  # Output length was reduced


def test_truncate_messages(agent):
    """Test when message truncation is needed."""

    # Fill the context with messages that will require truncation
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add several messages that will need truncation
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User message {i+1} " + "X" * 8_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant reply {i+1}")
        )

    orig_msg_len = len(agent.message_history[1].content)
    # Call the method
    hist, output_len = agent._prep_llm_messages("Final message")

    # Check that early messages were truncated
    assert len(hist) == 8  # All messages still present
    assert len(hist[1].content) < orig_msg_len
    # First user message truncated
    assert "Contents truncated" in hist[1].content
    assert output_len >= MIN_OUTPUT_TOKENS  # At least min_output_tokens


@pytest.fixture
def agent_drop_turns():
    """Create a ChatAgent with drop_turns strategy for testing."""
    config = ChatAgentConfig(
        system_message="System message",
        context_overflow_strategy="drop_turns",
        llm=OpenAIGPTConfig(
            # Small context for testing truncation
            chat_context_length=CHAT_CONTEXT_LENGTH,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            min_output_tokens=MIN_OUTPUT_TOKENS,
        ),
    )
    agent = ChatAgent(config)

    # Create a mock parser that counts tokens as characters for simplicity
    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            else:
                return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    # Create a mock LLM that returns a fixed context length
    class MockLLM:
        def chat_context_length(self):
            return CHAT_CONTEXT_LENGTH

        def supports_functions_or_tools(self):
            return False

    agent.llm = MockLLM()

    agent.init_message_history()
    return agent


def test_drop_turns_strategy(agent_drop_turns):
    """Test when drop_turns strategy is used to handle context overflow."""
    agent = agent_drop_turns

    # Fill the context with messages that will require dropping turns
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add several complete turns that will need to be dropped
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User message {i+1} " + "X" * 8_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant reply {i+1}")
        )

    orig_hist_len = len(agent.message_history)
    # Call the method
    hist, output_len = agent._prep_llm_messages("Final message")

    # Check that turns were dropped (fewer messages than original)
    assert len(hist) < orig_hist_len
    # System message should still be present
    assert hist[0].role == Role.SYSTEM
    assert hist[0].content == "System message"
    # The last user message should be present
    assert hist[-1].role == Role.USER
    assert hist[-1].content == "Final message"
    # Check alternating pattern is preserved
    for i in range(1, len(hist) - 1, 2):
        assert hist[i].role == Role.USER
        assert hist[i + 1].role == Role.ASSISTANT
    assert output_len >= MIN_OUTPUT_TOKENS


def test_drop_turns_preserves_last_turn(agent_drop_turns):
    """Test that drop_turns preserves the system message and last turn."""
    agent = agent_drop_turns

    # Set up history with multiple turns
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="System message")]

    # Add turns with large content that will force dropping
    for i in range(4):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"User {i+1} " + "Y" * 6_000)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"Assistant {i+1}")
        )

    # Call the method with a final message
    hist, output_len = agent._prep_llm_messages("Final user message")

    # System message must be preserved
    assert hist[0].role == Role.SYSTEM
    # Last message must be the final user message
    assert hist[-1].content == "Final user message"
    # No message should contain "Contents truncated" (we drop, not truncate)
    for msg in hist:
        assert "Contents truncated" not in msg.content


def test_drop_turns_accounts_for_buffer():
    """
    Test that drop_turns loop accounts for CHAT_HISTORY_BUFFER.

    This is a regression test for a P1 bug where the loop would exit when:
        tokens <= context - min_output_tokens
    But then output_len = context - tokens - CHAT_HISTORY_BUFFER could go
    negative, causing spurious errors.

    The fix ensures the loop continues until there's room for both
    min_output_tokens AND CHAT_HISTORY_BUFFER.
    """
    # CHAT_HISTORY_BUFFER is 300 in the code
    # We need to create a scenario where history is in the "danger zone":
    # between (context - min_output - buffer) and (context - min_output)
    #
    # With context=16000, min_output=50, buffer=300:
    # - Old buggy threshold: 16000 - 50 = 15950
    # - Fixed threshold: 16000 - 50 - 300 = 15650
    # - Danger zone: 15650 < tokens <= 15950

    config = ChatAgentConfig(
        system_message="S" * 100,  # 100 tokens
        context_overflow_strategy="drop_turns",
        llm=OpenAIGPTConfig(
            chat_context_length=16_000,
            max_output_tokens=1000,
            min_output_tokens=50,
        ),
    )
    agent = ChatAgent(config)

    class MockParser:
        def num_tokens(self, text: str | LLMMessage):
            if isinstance(text, str):
                return len(text)
            return len(text.content)

        def truncate_tokens(self, text, tokens, warning=""):
            return text[:tokens] + warning

    agent.parser = MockParser()

    class MockLLM:
        def chat_context_length(self):
            return 16_000

        def supports_functions_or_tools(self):
            return False

    agent.llm = MockLLM()
    agent.init_message_history()

    # Create history that lands in the danger zone after some turns
    # System msg = 100 tokens
    # We want total around 15800-15900 tokens (in danger zone)
    # Add turns that will require the buffer-aware loop to drop them
    agent.message_history = [LLMMessage(role=Role.SYSTEM, content="S" * 100)]

    # Add turns: each turn is ~5000 tokens (USER 4980 + ASSISTANT 20)
    # 3 turns = ~15000 + system 100 = ~15100
    # Final message ~800 = ~15900 total (in danger zone)
    for i in range(3):
        agent.message_history.append(
            LLMMessage(role=Role.USER, content=f"U{i}" + "X" * 4978)
        )
        agent.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=f"A{i}" + "Y" * 18)
        )

    # This should NOT raise an error - the fix ensures we drop enough turns
    # to accommodate both min_output_tokens AND CHAT_HISTORY_BUFFER
    hist, output_len = agent._prep_llm_messages("Z" * 800)

    # output_len must be positive and at least min_output_tokens
    assert output_len >= 50, f"output_len={output_len} should be >= 50"
    # History should have been compressed
    assert hist[0].role == Role.SYSTEM
    assert hist[-1].role == Role.USER
