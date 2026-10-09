"""Offline tests through real Langroid tasks and MCP client/server dispatch."""

import asyncio
import importlib.util
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
    prompts: list[str] = []

    def record(prompt: str) -> str:
        prompts.append(prompt)
        return "the model should never be consulted here"

    with pytest.raises(ValueError, match="Expected one of each"):
        await example.research("Research", MockLMConfig(response_fn=record), mcp)
    assert calls == []
    # The "before model" half of this test's name: a recording model, so that
    # consulting the model before raising cannot satisfy the assertions.
    # `MockLMConfig()` alone would have passed either way.
    assert prompts == []
    assert lifecycle == ["started", "closed"]


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
