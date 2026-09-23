"""Offline tests through real Langroid tasks and MCP client/server dispatch."""

import asyncio
import importlib.util
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import FastMCP

from langroid.language_models.mock_lm import MockLMConfig

EXAMPLE = Path(__file__).parents[2] / "examples/mcp/baizhi-research.py"
SPEC = importlib.util.spec_from_file_location("baizhi_example", EXAMPLE)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

ServerFixture = tuple[FastMCP, list[tuple[str, str]], list[str]]


@pytest.fixture
def server() -> ServerFixture:
    """Use synthetic schemas and observe the actual server lifespan."""
    lifecycle: list[str] = []

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
        lifecycle.append("started")
        try:
            yield {}
        finally:
            lifecycle.append("closed")

    mcp = FastMCP("Synthetic research server", lifespan=lifespan)
    calls: list[tuple[str, str]] = []

    for name in (*sorted(example.RESEARCH_TOOLS), "unrelated_tool"):

        def make_tool(tool_name: str) -> Callable[[str], Awaitable[str]]:
            async def tool(value: str) -> str:
                calls.append((tool_name, value))
                return f"Observed {value} at https://example.org/source"

            return tool

        mcp.tool(name=name)(make_tool(name))
    return mcp, calls, lifecycle


def mock_http_edge(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.AsyncClient]:
    """Replace only network I/O; retain the real HTTP and MCP clients."""
    real_client = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(example.httpx, "AsyncClient", create_client)
    return clients


@pytest.mark.parametrize("key", [None, "", "  "])
def test_missing_key(monkeypatch: pytest.MonkeyPatch, key: str | None) -> None:
    if key is None:
        monkeypatch.delenv("BAIZHI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("BAIZHI_API_KEY", key)
    with pytest.raises(ValueError, match="Set BAIZHI_API_KEY"):
        example.baizhi_transport()


@pytest.mark.asyncio
async def test_auth_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAIZHI_API_KEY", " synthetic-test-key ")
    transport = example.baizhi_transport()
    assert str(transport.url) == example.BAIZHI_MCP_URL
    assert transport.headers == {"Authorization": "Bearer synthetic-test-key"}
    client = transport.httpx_client_factory(headers=transport.headers)
    async with client:
        assert client.headers["Authorization"] == "Bearer synthetic-test-key"
        assert not client.follow_redirects
        assert not client.trust_env


@pytest.mark.asyncio
async def test_real_task_calls_all_three_tools(server: ServerFixture) -> None:
    mcp, calls, lifecycle = server
    names = sorted(example.RESEARCH_TOOLS)
    responses = iter(
        [json.dumps({"request": name, "value": name}) for name in names]
        + ["Evidence: https://example.org/source"]
    )
    result = await example.research(
        "Research synthetic pages",
        MockLMConfig(response_fn=lambda _: next(responses)),
        mcp,
        turns=12,
    )
    assert result is not None and "https://example.org/source" in result.content
    assert calls == [(name, name) for name in names]
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
async def test_plain_answer_finishes_without_user_input(
    server: ServerFixture,
) -> None:
    mcp, calls, lifecycle = server
    model = MockLMConfig(default_response="No supporting evidence.")
    result = await example.research("Research", model, mcp)
    assert result is not None and result.content == "No supporting evidence."
    assert calls == []
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
async def test_unrelated_tool_not_dispatched(server: ServerFixture) -> None:
    mcp, calls, lifecycle = server
    await example.research(
        "Try a tool outside the allowlist",
        MockLMConfig(
            default_response=json.dumps(
                {"request": "unrelated_tool", "value": "must not run"}
            )
        ),
        mcp,
        turns=3,
    )
    assert calls == []
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
async def test_missing_tool_stops_before_model(server: ServerFixture) -> None:
    mcp, calls, lifecycle = server
    mcp.remove_tool("web_extract")
    with pytest.raises(ValueError, match="Expected one of each"):
        await example.research("Research", MockLMConfig(), mcp)
    assert calls == []
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,turns", [(" ", 12), ("Research", 0)])
async def test_invalid_input_before_connection(query: str, turns: int) -> None:
    # An invalid URL would fail differently if a connection were attempted.
    with pytest.raises(ValueError, match="nonempty query and positive turns"):
        await example.research(query, MockLMConfig(), "invalid-url", turns)


@pytest.mark.asyncio
async def test_session_closes_on_model_failure(server: ServerFixture) -> None:
    mcp, calls, lifecycle = server

    async def fail(_: str) -> str:
        raise RuntimeError("synthetic model failure")

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        model = MockLMConfig(response_fn_async=fail)
        await example.research("Research", model, mcp)
    assert calls == []
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
async def test_session_closes_when_cancelled_during_tool(
    server: ServerFixture,
) -> None:
    mcp, calls, lifecycle = server
    tool_started = asyncio.Event()
    mcp.remove_tool("web_scrape")

    @mcp.tool(name="web_scrape")
    async def wait_for_cancellation(value: str) -> str:
        calls.append(("web_scrape", value))
        tool_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    task = asyncio.create_task(
        example.research(
            "Research",
            MockLMConfig(
                default_response=json.dumps(
                    {"request": "web_scrape", "value": "synthetic-page"}
                )
            ),
            mcp,
        )
    )
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=10)
        assert lifecycle == ["started"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert calls == [("web_scrape", "synthetic-page")]
    assert lifecycle == ["started", "closed"]


@pytest.mark.asyncio
async def test_http_client_respects_timeout() -> None:
    timeout = httpx.Timeout(1.0)
    async with example.http_client(timeout=timeout) as client:
        assert client.timeout == timeout


@pytest.mark.asyncio
async def test_http_transport_and_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive real Streamable HTTP protocol over synthetic HTTP responses."""
    monkeypatch.setenv("BAIZHI_API_KEY", "synthetic-http-key")
    methods: list[str] = []
    calls: list[dict[str, Any]] = []
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer synthetic-http-key"
        assert str(request.url) == example.BAIZHI_MCP_URL
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "GET":
            return httpx.Response(405)
        body = json.loads(request.content)
        method = body["method"]
        methods.append(method)
        if "id" not in body:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synthetic", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    }
                    for name in sorted(example.RESEARCH_TOOLS)
                ]
            }
        elif method == "tools/call":
            calls.append(body["params"])
            observation = "Source https://example.org/source"
            result = {
                "content": [{"type": "text", "text": observation}],
                "isError": False,
            }
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
            headers={"mcp-session-id": "synthetic-session"},
        )

    clients = mock_http_edge(monkeypatch, handle)
    responses = iter(
        [
            json.dumps({"request": "web_extract", "value": "synthetic-page"}),
            "Evidence: https://example.org/source",
        ]
    )
    result = await example.research(
        "Research",
        MockLMConfig(response_fn=lambda _: next(responses)),
        example.baizhi_transport(),
    )
    assert result is not None and "https://example.org/source" in result.content
    assert {"initialize", "tools/list", "tools/call"} <= set(methods)
    arguments = {"value": "synthetic-page"}
    assert calls == [{"name": "web_extract", "arguments": arguments}]
    assert any(request.method == "DELETE" for request in requests)
    assert clients and all(client.is_closed for client in clients)
    assert all(not client.follow_redirects for client in clients)
    assert all(not client.trust_env for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [307, 401])
async def test_http_redirect_or_auth_failure_closes_client(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv("BAIZHI_API_KEY", "synthetic-http-key")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"location": "https://other.example/mcp"}
        return httpx.Response(status, headers=headers)

    clients = mock_http_edge(monkeypatch, handle)
    transport = example.baizhi_transport()
    with pytest.raises(Exception):
        async with transport.connect_session() as session:
            await session.initialize()
    # The real transport invoked the custom factory and performed HTTP I/O.
    # The redirect target receives no request, even though the SDK requested
    # redirects when it invoked the factory.
    assert len(requests) == 1
    assert str(requests[0].url) == example.BAIZHI_MCP_URL
    assert requests[0].headers["Authorization"] == "Bearer synthetic-http-key"
    assert clients and all(client.is_closed for client in clients)
    assert all(not client.follow_redirects for client in clients)
