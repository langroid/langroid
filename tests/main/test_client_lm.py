"""Tests for ClientLM - MCP client-based language model."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from langroid.language_models.base import LanguageModel, LLMMessage
from langroid.language_models.client_lm import ClientLM, ClientLMConfig


class MockTextContent:
    """Mock TextContent returned by MCP sampling."""

    def __init__(self, text: str):
        self.text = text


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
    config = ClientLMConfig(context=mock_context)
    return ClientLM(config)


def test_client_lm_config():
    """Test ClientLMConfig initialization."""
    config = ClientLMConfig()
    assert config.type == "client"
    assert config.context is None
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
    mock_context.sample.return_value = MockTextContent(
        "Paris is the capital of France."
    )

    # Test with string input
    response = await client_lm.achat("What is the capital of France?")

    assert response.message == "Paris is the capital of France."
    assert response.cached is False

    # Verify sample was called correctly
    mock_context.sample.assert_called_once()
    call_args = mock_context.sample.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    assert call_args.kwargs["system_prompt"] is None
    assert call_args.kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_achat_with_messages(client_lm, mock_context):
    """Test async chat with message list."""
    # Setup mock response
    mock_context.sample.return_value = MockTextContent("Hello! I can help with that.")

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
    expected_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Can you help me?"},
    ]
    assert call_args.kwargs["messages"] == expected_messages
    assert call_args.kwargs["system_prompt"] == "You are a helpful assistant."
    assert call_args.kwargs["max_tokens"] == 500


@pytest.mark.asyncio
async def test_achat_no_context_error():
    """Test that achat raises error when no context is available."""
    config = ClientLMConfig()  # No context set
    lm = ClientLM(config)

    with pytest.raises(RuntimeError, match="No MCP context available for sampling"):
        await lm.achat("Hello")


@pytest.mark.asyncio
async def test_agenerate(client_lm, mock_context):
    """Test async generate functionality."""
    # Setup mock response
    mock_context.sample.return_value = MockTextContent("Generated response")

    response = await client_lm.agenerate("Generate something", max_tokens=200)

    assert response.message == "Generated response"

    # Verify it calls achat internally
    call_args = mock_context.sample.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "user", "content": "Generate something"}
    ]
    assert call_args.kwargs["max_tokens"] == 200


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
    mock_context.sample.return_value = MockTextContent("Response")

    await client_lm.achat("Hello")

    call_args = mock_context.sample.call_args
    assert call_args.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_handle_different_response_formats(client_lm, mock_context):
    """Test handling of different response formats from MCP."""

    # Test with object that has 'content' attribute
    class ContentObject:
        content = "Content response"

    mock_context.sample.return_value = ContentObject()
    response = await client_lm.achat("Test")
    assert response.message == "Content response"

    # Test with object that needs str() conversion
    class CustomObject:
        def __str__(self):
            return "String representation"

    mock_context.sample.return_value = CustomObject()
    response = await client_lm.achat("Test")
    assert response.message == "String representation"
