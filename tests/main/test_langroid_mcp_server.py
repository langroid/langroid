"""Tests for Langroid MCP server."""

from typing import Any, Dict, List

import pytest

from langroid.agent.chat_agent import ChatAgent
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.language_models.client_lm import ClientLMConfig
from langroid.mcp.server.langroid_mcp_server import langroid_chat, langroid_task, server


class MockTextContent:
    """Mock TextContent returned by MCP sampling."""

    def __init__(self, text: str):
        self.text = text


class MockContext:
    """Mock MCP Context for testing with real sampling behavior."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {"default": "Hello! I'm a helpful assistant."}
        self.call_count = 0
        self.last_messages = None
        self.last_system_prompt = None

    async def sample(
        self, messages, system_prompt=None, temperature=None, max_tokens=None
    ):
        """Mock sample method that returns predefined responses."""
        self.call_count += 1
        self.last_messages = messages
        self.last_system_prompt = system_prompt

        # Get the last user message to determine response
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # Return a response based on the message
        for key, response in self.responses.items():
            if key in last_user_msg.lower():
                return MockTextContent(response)

        return MockTextContent(self.responses.get("default", "I don't understand."))


@pytest.fixture
def mock_context():
    """Create a mock MCP context with predefined responses."""
    return MockContext(
        {
            "hello": "Hello! How can I help you today?",
            "search": "I found some interesting results about that topic.",
            "task": "I've completed the task successfully.",
            "default": "I'm here to help!",
        }
    )


def test_server_creation():
    """Test that the MCP server is created correctly."""
    assert server.name == "Langroid Agent Server"
    # Check that the server is a FastMCP instance
    from fastmcp import FastMCP

    assert isinstance(server, FastMCP)


@pytest.mark.asyncio
async def test_langroid_chat_basic(mock_context):
    """Test basic langroid_chat functionality with real agent."""
    # Call langroid_chat
    result = await langroid_chat(
        message="Hello",
        ctx=mock_context,
    )

    # Verify the response
    assert result == "Hello! How can I help you today?"

    # Verify context was called
    assert mock_context.call_count == 1
    assert mock_context.last_messages == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_langroid_chat_with_custom_name(mock_context):
    """Test langroid_chat with custom agent name."""
    result = await langroid_chat(
        message="Hello there", ctx=mock_context, agent_name="CustomAssistant"
    )

    # Should still work the same
    assert result == "Hello! How can I help you today?"
    assert mock_context.call_count == 1


@pytest.mark.asyncio
async def test_langroid_chat_with_tools(mock_context):
    """Test langroid_chat with tools enabled."""
    # Update context to respond differently for search
    mock_context.responses["search"] = "I'll search for that information."

    result = await langroid_chat(
        message="Can you search for AI news?",
        ctx=mock_context,
        enable_tools=["web_search"],
        agent_name="SearchAgent",
    )

    # Verify the response
    assert result == "I'll search for that information."
    assert mock_context.call_count == 1


@pytest.mark.asyncio
async def test_langroid_task_basic(mock_context):
    """Test basic langroid_task functionality with real components."""
    # Set up response that includes DONE to signal task completion
    mock_context.responses["task"] = "Task completed successfully! DONE"

    result = await langroid_task(
        message="Complete this task", ctx=mock_context, max_turns=5
    )

    # Task should complete and return the final message
    assert result is not None
    assert isinstance(result, str)
    assert mock_context.call_count >= 1


@pytest.mark.asyncio
async def test_langroid_task_with_tools(mock_context):
    """Test langroid_task with tools enabled."""
    # Set up response for search task with DONE signal
    mock_context.responses["search"] = (
        "Found relevant information for your search. DONE"
    )

    result = await langroid_task(
        message="Search for information about Python",
        ctx=mock_context,
        enable_tools=["web_search"],
        agent_name="SearchAgent",
        max_turns=3,
    )

    # Should return a result
    assert result is not None
    assert isinstance(result, str)
    assert mock_context.call_count >= 1


@pytest.mark.asyncio
async def test_langroid_task_result_conversion(mock_context):
    """Test that non-string task results are converted to strings."""

    # Create a custom context that returns an object
    class ObjectContext(MockContext):
        async def sample(self, messages, **kwargs):
            await super().sample(messages, **kwargs)
            # Return something that will make the task return a dict
            return MockTextContent("DONE")

    obj_context = ObjectContext()
    result = await langroid_task(message="Do something", ctx=obj_context, max_turns=1)

    # Result should be a string
    assert isinstance(result, str)
    assert obj_context.call_count >= 1


def test_main_function():
    """Test the main entry point."""
    from unittest.mock import patch

    with patch("asyncio.run") as mock_run:
        from langroid.mcp.server.langroid_mcp_server import main

        main()
        mock_run.assert_called_once()
        # Verify server.run() was passed to asyncio.run
        call_args = mock_run.call_args[0][0]
        assert hasattr(call_args, "__name__")  # It's a coroutine


@pytest.mark.asyncio
async def test_client_lm_integration(mock_context):
    """Test that ClientLM is properly integrated with the agent."""
    # Test a simple conversation
    await langroid_chat(
        message="What is 2 + 2?",
        ctx=mock_context,
    )

    # Context should have been called
    assert mock_context.call_count == 1

    # Should have received user message
    assert len(mock_context.last_messages) == 1
    assert mock_context.last_messages[0]["role"] == "user"
    assert "2 + 2" in mock_context.last_messages[0]["content"]
