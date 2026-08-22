"""Tests for cooperative wall-clock limits in Task run loops."""

from collections.abc import Iterator

import pytest

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.pydantic_v1 import BaseModel


class _ExpectedResult(BaseModel):
    """Structured result used to exercise strict decoding."""

    value: int


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


def test_negative_wall_clock_budget_keeps_unlimited_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative budget is unlimited and does not consult the clock."""
    monkeypatch.setattr(
        "langroid.agent.task.monotonic",
        lambda: pytest.fail("unlimited runs must not read the budget clock"),
    )
    task = lr.Task(_looping_agent(), interactive=False)

    result = task.run("start", turns=1, max_time=-1)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.FIXED_TURNS


def test_nan_wall_clock_budget_is_rejected() -> None:
    """A NaN budget is invalid rather than silently becoming unlimited."""
    task = lr.Task(_looping_agent(), interactive=False)

    with pytest.raises(ValueError, match="max_time must not be NaN"):
        task.run("start", turns=1, max_time=float("nan"))


@pytest.mark.asyncio
async def test_async_nan_wall_clock_budget_is_rejected() -> None:
    """The async loop rejects NaN budgets identically to the sync loop."""
    task = lr.Task(_looping_agent(), interactive=False)

    with pytest.raises(ValueError, match="max_time must not be NaN"):
        await task.run_async("start", turns=1, max_time=float("nan"))


def test_run_allows_several_under_budget_steps_before_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget is checked after every step, not only the first one."""
    readings = _clock([10.0, 10.1, 10.2, 10.6])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT
    with pytest.raises(StopIteration):
        next(readings)


def test_completion_wins_when_step_also_exhausts_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed step returns DONE even when it crosses the deadline."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: "DONE: finished"))
    )
    task = lr.Task(agent, interactive=False)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.content.endswith("finished")
    assert result.metadata.status == lr.StatusCode.DONE


@pytest.mark.asyncio
async def test_async_completion_wins_when_step_also_exhausts_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async loop also gives task completion precedence over timeout."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: "DONE: finished"))
    )
    task = lr.Task(agent, interactive=False)

    result = await task.run_async("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.content.endswith("finished")
    assert result.metadata.status == lr.StatusCode.DONE


def test_timeout_skips_unbudgeted_strict_decode_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout does not issue another LLM call to repair typed output."""
    calls: list[str] = []

    def response(message: str) -> str:
        calls.append(message)
        return "not structured"

    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=response)))
    monkeypatch.setattr(agent, "_json_schema_available", lambda: True)
    task = lr.Task(agent, interactive=False)

    result = task.run(
        "start",
        turns=10,
        max_time=0.5,
        return_type=_ExpectedResult,
    )

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_timeout_skips_unbudgeted_strict_decode_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async timeout path also skips the strict-decoding LLM retry."""
    calls: list[str] = []

    def response(message: str) -> str:
        calls.append(message)
        return "not structured"

    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=response)))
    monkeypatch.setattr(agent, "_json_schema_available", lambda: True)
    task = lr.Task(agent, interactive=False)

    result = await task.run_async(
        "start",
        turns=10,
        max_time=0.5,
        return_type=_ExpectedResult,
    )

    assert result is None
    assert len(calls) == 1


def test_delegated_subtask_step_is_followed_by_budget_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed delegated step can exhaust the parent task's budget."""
    readings = _clock([0.0, 0.1, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    parent_agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(response_fn=lambda _: "TO[child]: delegate"))
    )
    parent = lr.Task(parent_agent, interactive=False, name="parent")
    child = lr.Task(
        _looping_agent(),
        interactive=False,
        name="child",
        done_if_response=[Entity.LLM],
    )
    parent.add_sub_task(child)

    result = parent.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.content == "continue"
    assert result.metadata.status == lr.StatusCode.TIMEOUT
