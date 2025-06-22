"""Tests for ClientLM - MCP client-based language model."""

from unittest.mock import AsyncMock

import pytest
from fastmcp.client.sampling import SamplingMessage

# Import the real TextContent from mcp.types
from mcp.types import TextContent

from langroid.language_models.base import LanguageModel, LLMMessage
from langroid.language_models.client_lm import ClientLM, ClientLMConfig


class MockContext:
    """Mock MCP Context for testing."""

    def __init__(self):
        self.sample = AsyncMock()


@pytest.fixture
def mock_context():
    """Create a mock MCP context."""
    return MockContext()


@pytest.fixture
def client_lm(mock_context):
    """Create a ClientLM instance with mock context."""
    config = ClientLMConfig()
    lm = ClientLM(config)
    lm.set_context(mock_context)  # Use set_context method
    return lm


def test_client_lm_config():
    """Test ClientLMConfig initialization."""
    config = ClientLMConfig()
    assert config.type == "client"
    assert config.chat_context_length == 1_000_000_000


def test_client_lm_factory_creation():
    """Test that ClientLM can be created via LanguageModel.create()."""
    config = ClientLMConfig()
    lm = LanguageModel.create(config)
    assert isinstance(lm, ClientLM)
    assert lm.config.type == "client"


@pytest.mark.asyncio
async def test_achat_basic(client_lm, mock_context):
    """Test basic async chat functionality."""
    # Setup mock response
    mock_context.sample.return_value = TextContent(
        type="text", text="Paris is the capital of France."
    )

    # Test with string input
    response = await client_lm.achat("What is the capital of France?")

    assert response.message == "Paris is the capital of France."
    assert response.cached is False

    # Verify sample was called correctly
    mock_context.sample.assert_called_once()
    call_args = mock_context.sample.call_args

    # Check that we have one message
    assert len(call_args.kwargs["messages"]) == 1

    # Check the message is a SamplingMessage
    msg = call_args.kwargs["messages"][0]
    assert isinstance(msg, SamplingMessage)
    assert msg.role == "user"
    assert msg.content.type == "text"
    assert msg.content.text == "What is the capital of France?"

    assert call_args.kwargs["system_prompt"] is None
    assert call_args.kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_achat_with_messages(client_lm, mock_context):
    """Test async chat with message list."""
    # Setup mock response
    mock_context.sample.return_value = TextContent(
        type="text", text="Hello! I can help with that."
    )

    # Test with message list
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Hello"),
        LLMMessage(role="assistant", content="Hi there!"),
        LLMMessage(role="user", content="Can you help me?"),
    ]

    response = await client_lm.achat(messages, max_tokens=500)

    assert response.message == "Hello! I can help with that."

    # Verify sample was called with correct format
    call_args = mock_context.sample.call_args

    # Check that we have three messages
    messages = call_args.kwargs["messages"]
    assert len(messages) == 3

    # Check each message is a SamplingMessage with correct content
    assert all(isinstance(msg, SamplingMessage) for msg in messages)

    assert messages[0].role == "user"
    assert messages[0].content.text == "Hello"

    assert messages[1].role == "assistant"
    assert messages[1].content.text == "Hi there!"

    assert messages[2].role == "user"
    assert messages[2].content.text == "Can you help me?"

    assert call_args.kwargs["system_prompt"] == "You are a helpful assistant."
    assert call_args.kwargs["max_tokens"] == 500


@pytest.mark.asyncio
async def test_achat_no_context_error():
    """Test that achat raises error when no context is available."""
    config = ClientLMConfig()  # No context set
    lm = ClientLM(config)

    with pytest.raises(RuntimeError, match="No MCP context available for sampling"):
        await lm.achat("Hello")


def test_sync_methods_not_implemented(client_lm):
    """Test that synchronous methods raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="only supports async"):
        client_lm.chat("Hello")

    with pytest.raises(NotImplementedError, match="only supports async"):
        client_lm.generate("Hello")


def test_streaming_not_supported(client_lm):
    """Test that streaming is not supported."""
    assert client_lm.get_stream() is False
    assert client_lm.set_stream(True) is False
    assert client_lm.get_stream() is False


@pytest.mark.asyncio
async def test_temperature_passed_through(client_lm, mock_context):
    """Test that temperature from config is passed to sampling."""
    client_lm.config.temperature = 0.7
    mock_context.sample.return_value = TextContent(type="text", text="Response")

    await client_lm.achat("Hello")

    call_args = mock_context.sample.call_args
    assert call_args.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_handle_different_response_formats(client_lm, mock_context):
    """Test handling of different response formats from MCP."""

    # Test with TextContent (the expected format)
    mock_context.sample.return_value = TextContent(
        type="text", text="Text content response"
    )
    response = await client_lm.achat("Test")
    assert response.message == "Text content response"

    # Test that ImageContent raises NotImplementedError
    from mcp.types import ImageContent

    mock_context.sample.return_value = ImageContent(
        type="image", mimeType="image/png", data="base64data"
    )

    with pytest.raises(NotImplementedError, match="ImageContent"):
        await client_lm.achat("Test")
