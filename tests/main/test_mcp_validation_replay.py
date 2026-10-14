"""A response-validation failure must not execute an MCP tool again."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock

import pytest
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent, Tool

from langroid.agent.tools.mcp import FastMCPClient


@asynccontextmanager
async def _counting_client(
    result: CallToolResult, output_schema: dict[str, Any] | None
) -> AsyncIterator[tuple[FastMCPClient, list[dict[str, Any]]]]:
    """Connect a real MCP session to an in-memory, state-mutating server."""
    server = Server("counting-server")
    calls: list[dict[str, Any]] = []

    @server.list_tools()  # type: ignore[no-untyped-call, misc]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="increment",
                inputSchema={"type": "object"},
                outputSchema=output_schema,
            )
        ]

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        assert name == "increment"
        calls.append(arguments)
        # A raw result lets the server reproduce malformed output without
        # server-side normalization repairing it before the client sees it.
        return result

    async with create_connected_server_and_client_session(server) as session:
        client = FastMCPClient("unused")
        # Only supply the connection holder: dispatch, transport, server
        # execution, and client-side output validation all run natively.
        client.client = cast(Any, SimpleNamespace(session=session))
        yield client, calls


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "structured_content, error_text",
    [
        (None, "has an output schema but did not return structured content"),
        ({"count": "invalid"}, "Invalid structured content returned"),
    ],
    ids=["missing-structured-content", "invalid-structured-content"],
)
async def test_output_validation_failure_does_not_repeat_tool(
    structured_content: dict[str, Any] | None, error_text: str
) -> None:
    """Even a malformed response may follow a completed server-side mutation."""
    result = CallToolResult(
        content=[TextContent(type="text", text="incremented")],
        structuredContent=structured_content,
    )
    async with _counting_client(result, OUTPUT_SCHEMA) as (client, calls):
        error = None
        try:
            await client.call_mcp_tool("increment", {"amount": 1})
        except RuntimeError as exc:
            error = exc

    assert calls == [{"amount": 1}]
    assert isinstance(error, RuntimeError)
    assert error_text in str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, output_schema, expected_text",
    [
        (
            CallToolResult(
                content=[TextContent(type="text", text="incremented")],
                structuredContent={"count": 1},
            ),
            OUTPUT_SCHEMA,
            "incremented",
        ),
        (
            CallToolResult(content=[], structuredContent={"count": 1}),
            OUTPUT_SCHEMA,
            "1",
        ),
        (
            CallToolResult(content=[TextContent(type="text", text="incremented")]),
            None,
            "incremented",
        ),
        (
            CallToolResult(
                content=[TextContent(type="text", text="mutation refused")],
                isError=True,
            ),
            OUTPUT_SCHEMA,
            "ERROR: Tool call failed - ['mutation refused']",
        ),
    ],
    ids=["structured-and-text", "structured-only", "text-only", "tool-error"],
)
async def test_tool_results_are_preserved_without_replay(
    result: CallToolResult,
    output_schema: dict[str, Any] | None,
    expected_text: str,
) -> None:
    """Valid output and protocol-level tool errors retain their conversion."""
    async with _counting_client(result, output_schema) as (client, calls):
        actual = await client.call_mcp_tool("increment", {"amount": 1})

    assert actual == (expected_text, [])
    assert calls == [{"amount": 1}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Tool increment has an output schema but did not return structured content",
        "unrelated session failure",
    ],
)
async def test_original_runtime_error_propagates(message: str) -> None:
    """Preserve the original exception instead of replacing it or retrying."""
    error = RuntimeError(message)
    session = SimpleNamespace(
        call_tool=AsyncMock(side_effect=error),
        send_request=AsyncMock(side_effect=AssertionError("tool was replayed")),
    )
    client = FastMCPClient("unused")
    client.client = cast(Any, SimpleNamespace(session=session))

    with pytest.raises(RuntimeError) as caught:
        await client.call_mcp_tool("increment", {"amount": 1})

    assert caught.value is error
    session.call_tool.assert_awaited_once_with("increment", {"amount": 1})
    session.send_request.assert_not_awaited()
