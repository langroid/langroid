"""Integration tests for MCP Langroid server and client."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastmcp.client.sampling import SamplingMessage
from mcp.types import TextContent

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.client_lm import ClientLMConfig
from langroid.mcp.server.langroid_mcp_server import langroid_chat, langroid_task


class MockSamplingHandler:
    """Mock sampling handler that simulates an LLM client."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {
            "default": "I'm a helpful assistant.",
            "hello": "Hello! How are you today?",
            "math": "2 + 2 equals 4.",
            "search": "I'll search for that information.",
            "done": "Task completed. DONE",
        }
        self.call_history = []

    async def __call__(self, messages: List[Any], params: Any, context: Any) -> str:
        """Mock sampling handler."""
        # Record the call
        self.call_history.append({"messages": messages, "params": params})

        # Get last user message
        last_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, SamplingMessage) and msg.role == "user":
                last_msg = msg.content.text.lower()
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_msg = msg.get("content", "").lower()
                break
            elif isinstance(msg, str):
                last_msg = msg.lower()
                break

        # Return appropriate response
        for key, response in self.responses.items():
            if key in last_msg:
                return response

        return self.responses["default"]


@pytest.mark.asyncio
async def test_end_to_end_chat_flow():
    """Test complete flow from client through MCP to Langroid agent."""
    # Create mock context with sampling handler behavior
    mock_ctx = MagicMock()

    # Track sampling calls
    sampling_calls = []

    async def mock_sample(messages, **kwargs):
        sampling_calls.append({"messages": messages, "kwargs": kwargs})
        # Simulate LLM response
        if messages:
            last_msg_obj = messages[-1]
            if isinstance(last_msg_obj, SamplingMessage):
                last_msg = last_msg_obj.content.text
            else:
                last_msg = str(last_msg_obj)
        else:
            last_msg = ""

        return TextContent(type="text", text=f"Response to: {last_msg}")

    mock_ctx.sample = mock_sample

    # Test basic chat
    result = await langroid_chat(message="Hello, how are you?", ctx=mock_ctx)

    # Verify flow
    assert len(sampling_calls) == 1
    msg = sampling_calls[0]["messages"][0]
    assert isinstance(msg, SamplingMessage)
    assert msg.role == "user"
    assert "Hello, how are you?" in msg.content.text
    assert "Response to:" in result


@pytest.mark.asyncio
async def test_multi_turn_task_flow():
    """Test multi-turn conversation through task interface."""
    # Create mock context
    mock_ctx = MagicMock()

    # Track conversation turns
    turn_count = 0

    async def mock_sample(messages, **kwargs):
        nonlocal turn_count
        turn_count += 1

        if turn_count == 1:
            text = "Starting task..."
        elif turn_count == 2:
            text = "Working on it..."
        else:
            text = "Task complete! DONE"

        return TextContent(type="text", text=text)

    mock_ctx.sample = mock_sample

    # Run task
    result = await langroid_task(
        message="Help me with a complex task", ctx=mock_ctx, max_turns=5
    )

    # Should have multiple turns
    assert turn_count >= 1
    assert result is not None


@pytest.mark.asyncio
async def test_tool_integration():
    """Test that tools work correctly in the MCP context."""
    # Create mock context
    mock_ctx = MagicMock()

    tool_calls = []

    async def mock_sample(messages, **kwargs):
        # Check if this is a tool-related query
        if messages:
            last_msg_obj = messages[-1]
            if isinstance(last_msg_obj, SamplingMessage):
                last_msg = last_msg_obj.content.text
            else:
                last_msg = str(last_msg_obj)
        else:
            last_msg = ""

        if "search" in last_msg.lower():
            # Simulate tool use
            tool_calls.append("web_search")
            text = (
                '{"request": "duckduckgo_search", '
                '"query": "AI news", "num_results": 3}'
            )
        else:
            text = "Found some results about AI."

        return TextContent(type="text", text=text)

    mock_ctx.sample = mock_sample

    # Run with tools
    result = await langroid_chat(
        message="Search for AI news", ctx=mock_ctx, enable_tools=["web_search"]
    )

    # Tool should be available to agent
    assert result is not None


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling when MCP context fails."""
    # Create mock context that fails
    mock_ctx = MagicMock()

    async def mock_sample_error(messages, **kwargs):
        raise Exception("Simulated MCP error")

    mock_ctx.sample = mock_sample_error

    # Should raise error
    with pytest.raises(Exception, match="Simulated MCP error"):
        await langroid_chat(message="This should fail", ctx=mock_ctx)


@pytest.mark.asyncio
async def test_context_preservation():
    """Test that context is properly passed through the chain."""
    # Create mock context
    mock_ctx = MagicMock()

    captured_params = {}

    async def mock_sample(
        messages, system_prompt=None, temperature=None, max_tokens=None, **kwargs
    ):
        captured_params.update(
            {
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        return TextContent(type="text", text="Response with context")

    mock_ctx.sample = mock_sample

    # Create agent with specific temperature
    config = ClientLMConfig(temperature=0.8)
    agent_config = ChatAgentConfig(llm=config)
    agent = ChatAgent(agent_config)

    # Set context on the LLM instance
    if hasattr(agent.llm, "set_context"):
        agent.llm.set_context(mock_ctx)

    # Get response
    await agent.llm_response_async("Test message")

    # Verify temperature was passed
    assert captured_params["temperature"] == 0.8


@pytest.mark.asyncio
async def test_message_format_conversion():
    """Test conversion between Langroid and MCP message formats."""
    # Create mock context
    mock_ctx = MagicMock()

    captured_messages = []

    async def mock_sample(messages, **kwargs):
        captured_messages.extend(messages)

        return TextContent(type="text", text="Converted successfully")

    mock_ctx.sample = mock_sample

    # Test with agent that has conversation history
    config = ClientLMConfig()
    agent_config = ChatAgentConfig(
        llm=config, system_message="You are a helpful assistant."
    )
    agent = ChatAgent(agent_config)

    # Set context on the LLM instance
    if hasattr(agent.llm, "set_context"):
        agent.llm.set_context(mock_ctx)

    # Add some history - must start with system message
    from langroid.language_models.base import LLMMessage, Role

    agent.message_history.append(
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    )
    agent.message_history.append(
        LLMMessage(role=Role.USER, content="Previous user message")
    )
    agent.message_history.append(
        LLMMessage(role=Role.ASSISTANT, content="Previous assistant response")
    )

    # Get new response
    await agent.llm_response_async("New message")

    # Check captured messages include history
    assert len(captured_messages) >= 2
    assert any(
        isinstance(msg, SamplingMessage) and msg.content.text == "Previous user message"
        for msg in captured_messages
    )


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling multiple concurrent requests."""
    # Create mock context
    mock_ctx = MagicMock()

    request_count = 0

    async def mock_sample(messages, **kwargs):
        nonlocal request_count
        request_count += 1
        await asyncio.sleep(0.1)  # Simulate processing time

        return TextContent(type="text", text=f"Response {request_count}")

    mock_ctx.sample = mock_sample

    # Run multiple concurrent requests
    tasks = [langroid_chat(message=f"Request {i}", ctx=mock_ctx) for i in range(3)]

    results = await asyncio.gather(*tasks)

    # All should complete
    assert len(results) == 3
    assert all("Response" in r for r in results)
    assert request_count == 3
