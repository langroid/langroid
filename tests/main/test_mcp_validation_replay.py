"""A response-validation failure must not execute an MCP tool again."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, TextContent, Tool

from langroid.agent.tools.mcp import FastMCPClient


@asynccontextmanager
async def _connected_session(server: Server[Any]) -> AsyncIterator[ClientSession]:
    """An initialized `ClientSession` speaking to `server` over memory streams.

    Inlined rather than imported: mcp 1.x shipped this as
    `mcp.shared.memory.create_connected_server_and_client_session`, which mcp
    2.x (fastmcp 4) removed. The primitives it is built from --
    `create_client_server_memory_streams`, `Server.run` and `ClientSession` --
    exist unchanged in both, so one local copy serves either version.
    """
    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                )
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()
                    yield session
            finally:
                tg.cancel_scope.cancel()


def _tool(name: str, output_schema: dict[str, Any] | None) -> Tool:
    """An `mcp.types.Tool`, built under whichever field names the SDK uses.

    MCP SDK v2 (``fastmcp>=4``) renamed `inputSchema`/`outputSchema` to
    `input_schema`/`output_schema` and keeps the old spellings only as
    validation ALIASES -- so the v1 keyword form is silently ignored by the
    constructor, while `model_validate` on the wire-format keys is accepted
    by both SDKs.
    """
    return Tool.model_validate(
        {
            "name": name,
            "inputSchema": {"type": "object"},
            "outputSchema": output_schema,
        }
    )


def _result(
    content: list[TextContent],
    structured_content: dict[str, Any] | None = None,
    is_error: bool = False,
) -> CallToolResult:
    """A `CallToolResult`, built under whichever field names the SDK uses.

    Same alias story as `_tool`: `structuredContent`/`isError` are v1 names
    that v2 accepts only through validation.
    """
    return CallToolResult.model_validate(
        {
            "content": content,
            "structuredContent": structured_content,
            "isError": is_error,
        }
    )


@asynccontextmanager
async def _counting_client(
    result: CallToolResult, output_schema: dict[str, Any] | None
) -> AsyncIterator[tuple[FastMCPClient, list[dict[str, Any]]]]:
    """Connect a real MCP session to an in-memory, state-mutating server."""
    calls: list[dict[str, Any]] = []
    tools = [_tool("increment", output_schema)]

    def record(name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        assert name == "increment"
        calls.append(arguments or {})
        # A raw result lets the server reproduce malformed output without
        # server-side normalization repairing it before the client sees it.
        return result

    if hasattr(Server, "list_tools"):
        # MCP SDK v1 (fastmcp 2.x/3.x): handlers registered by decorator.
        server = Server("counting-server")

        @server.list_tools()  # type: ignore[no-untyped-call, misc, attr-defined]
        async def list_tools() -> list[Tool]:
            return tools

        @server.call_tool()  # type: ignore[misc, attr-defined]
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            return record(name, arguments)

    else:
        # MCP SDK v2 (fastmcp 4): handlers passed to the constructor, and
        # given a request context plus typed params instead of bare values.
        from mcp.types import ListToolsResult

        async def on_list_tools(ctx: Any, params: Any) -> Any:
            return ListToolsResult(tools=tools)

        async def on_call_tool(ctx: Any, params: Any) -> CallToolResult:
            return record(params.name, params.arguments)

        server = Server(
            "counting-server",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )

    async with _connected_session(server) as session:
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
    result = _result([TextContent(type="text", text="incremented")], structured_content)
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
            _result([TextContent(type="text", text="incremented")], {"count": 1}),
            OUTPUT_SCHEMA,
            "incremented",
        ),
        (
            _result([], {"count": 1}),
            OUTPUT_SCHEMA,
            "1",
        ),
        (
            _result([TextContent(type="text", text="incremented")]),
            None,
            "incremented",
        ),
        (
            _result([TextContent(type="text", text="mutation refused")], is_error=True),
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
