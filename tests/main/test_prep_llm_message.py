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

    # Call the method
    hist, output_len = agent._prep_llm_messages("Final message")

    # Check that early messages were truncated
    assert len(hist) == 8  # All messages still present
    assert len(hist[1].content) < len(
        agent.message_history[1].content
    )  # First user message truncated
    assert "Contents truncated" in hist[1].content
    assert output_len >= MIN_OUTPUT_TOKENS  # At least min_output_tokens
