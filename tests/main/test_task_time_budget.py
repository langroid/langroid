"""Tests for cooperative wall-clock limits in Task run loops."""

from collections.abc import Iterator

import pytest

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.language_models.mock_lm import MockLMConfig


def _clock(values: list[float]) -> Iterator[float]:
    """Yield deterministic monotonic-clock readings."""
    yield from values


def _looping_agent() -> ChatAgent:
    """Build an offline agent that keeps producing valid responses."""
    return ChatAgent(
        ChatAgentConfig(
            llm=MockLMConfig(response_fn=lambda _: "continue"),
        )
    )


def test_run_stops_at_wall_clock_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync loop reports TIMEOUT after a completed step exhausts its budget."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT


@pytest.mark.asyncio
async def test_run_async_stops_at_wall_clock_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async loop applies the same cooperative time budget as sync."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    result = await task.run_async("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT


def test_zero_wall_clock_budget_keeps_existing_unlimited_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero budget remains the backwards-compatible unlimited default."""
    monkeypatch.setattr(
        "langroid.agent.task.monotonic",
        lambda: pytest.fail("unlimited runs must not read the budget clock"),
    )
    task = lr.Task(_looping_agent(), interactive=False)

    result = task.run("start", turns=1, max_time=0)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.FIXED_TURNS
