"""Tests for cooperative wall-clock limits in Task run loops."""

import asyncio
import time
from collections.abc import Iterator
from typing import Any

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


def test_run_stops_exactly_at_wall_clock_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elapsed time equal to the budget is already timed out."""
    readings = _clock([0.0, 0.5])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT


def test_run_budget_includes_restart_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync budget starts before recursive restart work."""
    now = 0.0
    calls = 0

    def response(_: str) -> str:
        nonlocal calls
        calls += 1
        return "continue"

    def reset() -> None:
        nonlocal now
        now = 1.0

    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: now)
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=response)))
    task = lr.Task(agent, interactive=False)
    monkeypatch.setattr(task, "reset_all_sub_tasks", reset)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT
    assert calls == 1


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


@pytest.mark.asyncio
async def test_run_async_budget_includes_restart_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async budget starts before recursive restart work."""
    now = 0.0
    calls = 0

    async def response(_: str) -> str:
        nonlocal calls
        calls += 1
        return "continue"

    def reset() -> None:
        nonlocal now
        now = 1.0

    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: now)
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn_async=response)))
    task = lr.Task(agent, interactive=False)
    monkeypatch.setattr(task, "reset_all_sub_tasks", reset)

    result = await task.run_async("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT
    assert calls == 1


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


@pytest.mark.parametrize("unlimited_budget", [0, -1])
def test_run_budget_is_local_to_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
    unlimited_budget: float,
) -> None:
    """A prior timed run cannot constrain a later unlimited run."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    timed_result = task.run("start", turns=10, max_time=0.5)
    assert isinstance(timed_result, ChatDocument)
    assert timed_result.metadata.status == lr.StatusCode.TIMEOUT

    monkeypatch.setattr(
        "langroid.agent.task.monotonic",
        lambda: pytest.fail("unlimited runs must not read the budget clock"),
    )
    unlimited_result = task.run("start", turns=1, max_time=unlimited_budget)

    assert isinstance(unlimited_result, ChatDocument)
    assert unlimited_result.metadata.status == lr.StatusCode.FIXED_TURNS


@pytest.mark.asyncio
@pytest.mark.parametrize("unlimited_budget", [0, -1])
async def test_run_async_budget_is_local_to_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
    unlimited_budget: float,
) -> None:
    """Async non-positive budgets stay unlimited after a timed invocation."""
    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    task = lr.Task(_looping_agent(), interactive=False)

    timed_result = await task.run_async("start", turns=10, max_time=0.5)
    assert isinstance(timed_result, ChatDocument)
    assert timed_result.metadata.status == lr.StatusCode.TIMEOUT

    monkeypatch.setattr(
        "langroid.agent.task.monotonic",
        lambda: pytest.fail("unlimited runs must not read the budget clock"),
    )
    unlimited_result = await task.run_async(
        "start",
        turns=1,
        max_time=unlimited_budget,
    )

    assert isinstance(unlimited_result, ChatDocument)
    assert unlimited_result.metadata.status == lr.StatusCode.FIXED_TURNS


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


def test_run_rechecks_budget_after_done_before_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deadline expiry inside done prevents another sync step."""
    now = 0.0
    calls = 0

    def response(_: str) -> str:
        nonlocal calls
        calls += 1
        return "continue"

    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: now)
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=response)))
    task = lr.Task(agent, interactive=False)
    original_done = task.done

    def done(*args: Any, **kwargs: Any) -> tuple[bool, lr.StatusCode]:
        nonlocal now
        result = original_done(*args, **kwargs)
        if not args and not kwargs:
            now = 1.0
        return result

    monkeypatch.setattr(task, "done", done)

    result = task.run("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT
    assert calls == 1


@pytest.mark.asyncio
async def test_run_async_rechecks_budget_after_done_before_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deadline expiry inside done prevents another async step."""
    now = 0.0
    calls = 0

    async def response(_: str) -> str:
        nonlocal calls
        calls += 1
        return "continue"

    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: now)
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn_async=response)))
    task = lr.Task(agent, interactive=False)
    original_done = task.done

    def done(*args: Any, **kwargs: Any) -> tuple[bool, lr.StatusCode]:
        nonlocal now
        result = original_done(*args, **kwargs)
        if not args and not kwargs:
            now = 1.0
        return result

    monkeypatch.setattr(task, "done", done)

    result = await task.run_async("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT
    assert calls == 1


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


def test_expired_done_response_skips_strict_decode_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry during local decoding prevents a sync repair call."""
    calls: list[str] = []

    def response(message: str) -> str:
        calls.append(message)
        return "DONE: not structured"

    readings = _clock([0.0, 0.1, 1.0])
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
async def test_async_expired_done_response_skips_strict_decode_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry during local decoding prevents an async repair call."""
    calls: list[str] = []

    def response(message: str) -> str:
        calls.append(message)
        return "DONE: not structured"

    readings = _clock([0.0, 0.1, 1.0])
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


def test_timeout_parses_valid_structured_response_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sync timeout parses valid typed output from its completed step."""
    calls = 0

    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig()))

    def response(_: str | ChatDocument | None = None) -> ChatDocument:
        nonlocal calls
        calls += 1
        return agent.response_template(
            Entity.LLM,
            content="structured result",
            content_any=_ExpectedResult(value=7),
        )

    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    monkeypatch.setattr(agent, "llm_response", response)
    task = lr.Task(agent, interactive=False)

    result = task.run(
        "start",
        turns=10,
        max_time=0.5,
        return_type=_ExpectedResult,
    )

    assert result == _ExpectedResult(value=7)
    assert calls == 1


@pytest.mark.asyncio
async def test_async_timeout_parses_valid_structured_response_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An async timeout parses valid typed output from its completed step."""
    calls = 0

    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig()))

    async def response(_: str | ChatDocument | None = None) -> ChatDocument:
        nonlocal calls
        calls += 1
        return agent.response_template(
            Entity.LLM,
            content="structured result",
            content_any=_ExpectedResult(value=7),
        )

    readings = _clock([0.0, 1.0])
    monkeypatch.setattr("langroid.agent.task.monotonic", lambda: next(readings))
    monkeypatch.setattr(agent, "llm_response_async", response)
    task = lr.Task(agent, interactive=False)

    result = await task.run_async(
        "start",
        turns=10,
        max_time=0.5,
        return_type=_ExpectedResult,
    )

    assert result == _ExpectedResult(value=7)
    assert calls == 1


def test_run_does_not_interrupt_step_that_overshoots_budget() -> None:
    """A slow sync response completes before the budget is enforced."""
    completed = False

    def response(_: str) -> str:
        nonlocal completed
        time.sleep(0.03)
        completed = True
        return "continue"

    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn=response)))
    task = lr.Task(agent, interactive=False)

    result = task.run("start", turns=10, max_time=0.01)

    assert completed
    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT


@pytest.mark.asyncio
async def test_run_async_does_not_cancel_step_that_overshoots_budget() -> None:
    """A slow async response completes before the budget is enforced."""
    completed = False

    async def response(_: str) -> str:
        nonlocal completed
        await asyncio.sleep(0.03)
        completed = True
        return "continue"

    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(response_fn_async=response)))
    task = lr.Task(agent, interactive=False)

    result = await task.run_async("start", turns=10, max_time=0.01)

    assert completed
    assert isinstance(result, ChatDocument)
    assert result.metadata.status == lr.StatusCode.TIMEOUT


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


@pytest.mark.asyncio
async def test_async_delegated_subtask_has_independent_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An async child completes without inheriting its parent's deadline."""
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

    result = await parent.run_async("start", turns=10, max_time=0.5)

    assert isinstance(result, ChatDocument)
    assert result.content == "continue"
    assert result.metadata.status == lr.StatusCode.TIMEOUT
