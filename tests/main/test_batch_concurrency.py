"""Offline contracts for opt-in rolling bounded task execution."""

import asyncio
import time
from typing import Any

import pytest

import langroid.agent.batch as batch
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import settings


def run_public(work: Any, inputs: Any = None, **kwargs: Any) -> list[Any]:
    options = dict(
        batch_size=None,
        sequential=False,
        stop_on_first_result=False,
        handle_exceptions=batch.ExceptionHandling.RAISE,
        output_map=lambda value: value,
        message_template="Offline batch test",
    )
    options.update(kwargs)
    return batch.run_batched_tasks(
        ["a", "b", "c"] if inputs is None else inputs, work, **options
    )


def test_fixed_batches_wait_for_slowest() -> None:
    """Record the old public path's barrier with event-controlled work."""
    observed = []
    b_finished = asyncio.Event()
    c_started = asyncio.Event()

    async def work(value: Any, index: int) -> Any:
        if index == 0:
            await asyncio.wait_for(b_finished.wait(), timeout=5)
            # Let runnable tasks advance; A remains inside the first batch.
            await asyncio.sleep(0)
            observed.append(c_started.is_set())
        elif index == 1:
            b_finished.set()
        else:
            c_started.set()
        return value

    assert run_public(work, batch_size=2) == ["a", "b", "c"]
    assert observed == [False]
    assert c_started.is_set()


def test_public_rolling_refills_before_slowest_finishes() -> None:
    """C must start while A is blocked, after B frees a slot."""
    a_started = asyncio.Event()
    c_started = asyncio.Event()
    finished = []

    async def work(value: Any, index: int) -> Any:
        if index == 0:
            a_started.set()
            await asyncio.wait_for(c_started.wait(), timeout=5)
        elif index == 1:
            await asyncio.wait_for(a_started.wait(), timeout=5)
        else:
            assert 0 not in finished
            c_started.set()
        finished.append(index)
        return value

    assert run_public(work, max_concurrency=2) == ["a", "b", "c"]
    assert finished[0] == 1
    assert finished.index(2) < finished.index(0)


@pytest.mark.parametrize("n,capacity", [(0, 2), (1, 1), (1, 5), (9, 1), (100, 3)])
def test_capacity_order_and_owned_tasks(
    n: int, capacity: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[asyncio.Task[Any]] = []
    original = asyncio.create_task
    peak_owned = 0
    active = 0
    peak_active = 0

    def track(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        nonlocal peak_owned
        task = original(coro, **kwargs)
        created.append(task)
        peak_owned = max(peak_owned, sum(not t.done() for t in created))
        return task

    monkeypatch.setattr(asyncio, "create_task", track)

    async def work(value: Any, index: int) -> Any:
        nonlocal active, peak_active
        active += 1
        peak_active = max(active, peak_active)
        try:
            await asyncio.sleep(0)
            return (value, index)
        finally:
            active -= 1

    inputs = [str(i) for i in range(n)]
    results = run_public(work, inputs, max_concurrency=capacity)
    assert results == list(zip(inputs, range(n)))
    assert peak_active == min(n, capacity)
    assert peak_owned == min(n, capacity)
    assert len(created) == n
    assert all(task.done() for task in created)
    assert active == 0


@pytest.mark.parametrize("policy", list(batch.ExceptionHandling))
@pytest.mark.parametrize("source", ["task", "mapper", "cancel", "self_cancel"])
def test_failure_policies(policy: batch.ExceptionHandling, source: str) -> None:
    mapped = []

    async def work(value: Any, index: int) -> Any:
        if index == 1:
            if source == "task":
                raise ValueError("failed")
            if source == "cancel":
                raise asyncio.CancelledError("failed")
            if source == "self_cancel":
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                await asyncio.sleep(0)
        return value

    def output_map(value: Any) -> Any:
        mapped.append(value)
        if source == "mapper" and value == "b":
            raise ValueError("failed")
        return value

    def run() -> list[Any]:
        return run_public(
            work,
            max_concurrency=2,
            handle_exceptions=policy,
            output_map=output_map,
        )

    error = asyncio.CancelledError if "cancel" in source else ValueError
    if policy == batch.ExceptionHandling.RAISE:
        with pytest.raises(error):
            run()
    else:
        result = run()
        assert result[0] == "a" and result[2] == "c"
        if policy == batch.ExceptionHandling.RETURN_NONE:
            assert result[1] is None
        else:
            assert isinstance(result[1], error)
        assert mapped == (["a", "b", "c"] if source == "mapper" else ["a", "c"])


@pytest.mark.parametrize("policy", list(batch.ExceptionHandling))
def test_success_values_are_mapped_once(policy: batch.ExceptionHandling) -> None:
    returned_error = ValueError("a legitimate result")
    values = [returned_error, None, "none", "ok"]
    mapped = []

    async def work(value: Any, index: int) -> Any:
        return values[index]

    def output_map(value: Any) -> Any:
        mapped.append(value)
        return None if value == "none" else ("mapped", value)

    result = run_public(
        work,
        [str(i) for i in range(4)],
        max_concurrency=2,
        handle_exceptions=policy,
        output_map=output_map,
    )
    assert result == [
        ("mapped", returned_error),
        ("mapped", None),
        None,
        ("mapped", "ok"),
    ]
    assert mapped == values


@pytest.mark.parametrize("policy", list(batch.ExceptionHandling))
def test_external_cancellation_cleans_only_owned_tasks(
    policy: batch.ExceptionHandling,
) -> None:
    async def scenario() -> None:
        ready = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleaning = asyncio.Event()
        started = []
        cleaned = []
        owned = []

        async def work(value: Any, index: int) -> Any:
            owned.append(asyncio.current_task())
            started.append(index)
            if len(started) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release_cleanup.wait()
                cleaned.append(index)

        unrelated = asyncio.create_task(asyncio.Event().wait())
        parent = asyncio.create_task(
            batch._process_rolling_async(
                ["a", "b", "c"],
                work,
                2,
                policy,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            parent.cancel()
            await asyncio.wait_for(cleaning.wait(), 5)
            # Repeated caller cancellation must not abandon async cleanup.
            parent.cancel()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 5)
            assert started == [0, 1]
            assert sorted(cleaned) == [0, 1]
            assert all(task is not None and task.done() for task in owned)
            assert not unrelated.done()
        finally:
            release_cleanup.set()
            parent.cancel()
            unrelated.cancel()
            await asyncio.gather(parent, unrelated, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(batch.ExceptionHandling))
def test_mapper_requested_parent_cancel_stops_refill(
    policy: batch.ExceptionHandling,
) -> None:
    started = []

    async def work(value: Any, index: int) -> Any:
        started.append(index)
        return value

    def output_map(value: Any) -> Any:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        return value

    with pytest.raises(asyncio.CancelledError):
        run_public(
            work, max_concurrency=1, handle_exceptions=policy, output_map=output_map
        )
    assert started == [0]


def test_same_wakeup_failures_are_drained_in_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-wakeup failures are raised in input-index order, not completion order.

    All three tasks finish before the scheduler observes any of them, and they
    are handed back in reverse order, so the lowest input index must still be
    the one that propagates.
    """
    original_wait = asyncio.wait
    owned = []

    async def reverse_wait(tasks: Any, **kwargs: Any) -> Any:
        done, pending = await original_wait(tasks, **kwargs)
        assert len(done) == len(tasks)
        # Deliberately give the scheduler completed tasks in reverse order.
        return list(reversed(list(tasks))), pending

    monkeypatch.setattr(asyncio, "wait", reverse_wait)

    async def work(value: Any, index: int) -> Any:
        owned.append(asyncio.current_task())
        raise ValueError(f"failure-{index}")

    with pytest.raises(ValueError, match="failure-0"):
        run_public(work, max_concurrency=3)
    assert len(owned) == 3
    assert all(task is not None and task.done() for task in owned)


@pytest.mark.parametrize("source", ["task", "mapper"])
def test_fatal_error_stops_refill_and_drains_siblings(source: str) -> None:
    async def scenario() -> None:
        ready = asyncio.Event()
        started = []
        cleaned = []
        owned = []

        async def work(value: Any, index: int) -> Any:
            started.append(index)
            owned.append(asyncio.current_task())
            try:
                if index == 0:
                    await ready.wait()
                    if source == "task":
                        raise ValueError("failed")
                    return value
                ready.set()
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.append(index)

        def output_map(value: Any) -> Any:
            raise ValueError("failed")

        with pytest.raises(ValueError, match="failed"):
            await asyncio.wait_for(
                batch._process_rolling_async(
                    ["a", "b", "c"],
                    work,
                    2,
                    output_map=output_map,
                ),
                5,
            )
        assert started == [0, 1]
        assert sorted(cleaned) == [0, 1]
        assert all(task is not None and task.done() for task in owned)

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("source", ["task", "mapper"])
@pytest.mark.parametrize("policy", list(batch.ExceptionHandling))
def test_process_control_exceptions_propagate(
    error: type[BaseException],
    source: str,
    policy: batch.ExceptionHandling,
) -> None:
    async def work(value: Any, index: int) -> Any:
        if source == "task":
            raise error()
        return value

    def output_map(value: Any) -> Any:
        raise error()

    # Await inline so pytest can catch the process-control exception before
    # it reaches the event loop's top-level task runner.
    async def scenario() -> None:
        with pytest.raises(error):
            await batch._process_rolling_async(
                ["a"],
                work,
                1,
                policy,
                output_map,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_control_exception_during_sibling_cleanup(
    error: type[BaseException],
) -> None:
    # The sibling sleeps rather than waiting forever, and the call is timed, so
    # a cleanup that fails to cancel siblings FAILS this test instead of
    # hanging the suite: the call would then take the whole sleep.
    sibling_sleep = 5.0

    async def scenario() -> None:
        ready = asyncio.Event()

        async def work(value: Any, index: int) -> Any:
            if index == 0:
                await ready.wait()
                raise ValueError("ordinary failure")
            try:
                ready.set()
                await asyncio.sleep(sibling_sleep)
            finally:
                raise error()

        started = time.monotonic()
        with pytest.raises(error):
            await batch._process_rolling_async(["a", "b"], work, 2)
        assert time.monotonic() - started < sibling_sleep / 2

    asyncio.run(scenario())


@pytest.mark.parametrize("entry", ["common", "generator", "clones"])
@pytest.mark.parametrize("items", [[], ["a"]])
@pytest.mark.parametrize(
    "options,error,match",
    [
        # `match` targets the validator's own wording rather than just the
        # parameter name: Python's "unexpected keyword argument
        # 'max_concurrency'" would otherwise satisfy these cases on any build
        # that lacks the parameter, making them vacuous regression guards.
        ({"max_concurrency": True}, TypeError, "not bool"),
        ({"max_concurrency": False}, TypeError, "not bool"),
        ({"max_concurrency": 1.5}, TypeError, "not float"),
        ({"max_concurrency": "2"}, TypeError, "not str"),
        ({"max_concurrency": 0}, ValueError, "must be a positive integer"),
        ({"max_concurrency": -2}, ValueError, "must be a positive integer"),
        ({"max_concurrency": 2, "batch_size": 2}, ValueError, "requires batch_size"),
        ({"max_concurrency": 2, "sequential": True}, ValueError, "requires batch_size"),
        (
            {"max_concurrency": 2, "stop_on_first_result": True},
            ValueError,
            "requires batch_size",
        ),
    ],
)
def test_validation_precedes_user_code(
    entry: str,
    items: list[str],
    options: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("user code ran before validation")

    class ForbiddenTask:
        name = "unused"
        clone = forbidden

    options = {"sequential": False, **options}
    with pytest.raises(error, match=match):
        if entry == "common":
            run_public(forbidden, items, **options)
        elif entry == "generator":
            batch.run_batch_task_gen(
                forbidden,
                items,
                input_map=forbidden,
                **options,
            )
        else:
            batch.run_batch_tasks(
                ForbiddenTask(),
                items,
                input_map=forbidden,
                **options,
            )


@pytest.mark.parametrize("entry", ["generator", "clones"])
def test_public_task_entrypoints(
    entry: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a_started = asyncio.Event()
    c_started = asyncio.Event()
    seen_tasks = []
    gen_indices = []
    budget_calls = []
    original_run = Task.run_async

    async def response(value: str) -> str:
        if value == "a":
            a_started.set()
            await asyncio.wait_for(c_started.wait(), 5)
        elif value == "b":
            await asyncio.wait_for(a_started.wait(), 5)
        else:
            c_started.set()
        return value.upper()

    async def record_run(self: Task, *args: Any, **kwargs: Any) -> Any:
        seen_tasks.append(self)
        budget_calls.append(kwargs.copy())
        return await original_run(self, *args, **kwargs)

    monkeypatch.setattr(Task, "run_async", record_run)
    config = ChatAgentConfig(
        name="Worker",
        vecdb=None,
        llm=MockLMConfig(response_fn_async=response),
    )
    template = Task(
        ChatAgent(config),
        name="Worker",
        interactive=False,
        done_if_response=[Entity.LLM],
    )
    original_config = config.model_dump()
    orig_quiet = settings.quiet

    def generate(index: int) -> Task:
        gen_indices.append(index)
        return template.clone(index)

    kwargs = dict(
        sequential=False, max_concurrency=2, turns=2, max_cost=0.01, max_tokens=100
    )
    results = (
        batch.run_batch_task_gen(generate, ["a", "b", "c"], **kwargs)
        if entry == "generator"
        else batch.run_batch_tasks(template, ["a", "b", "c"], **kwargs)
    )
    assert all(isinstance(result, ChatDocument) for result in results)
    assert [result.content for result in results] == ["A", "B", "C"]
    assert [task.name for task in seen_tasks] == ["Worker-0", "Worker-1", "Worker-2"]
    assert len({id(task.agent.config) for task in seen_tasks}) == 3
    assert all(not task.agent.config.show_stats for task in seen_tasks)
    assert config.model_dump() == original_config
    assert template.agent.message_history == []
    assert settings.quiet == orig_quiet
    assert budget_calls == [dict(turns=2, max_cost=0.01, max_tokens=100)] * 3
    if entry == "generator":
        assert gen_indices == [0, 1, 2]


@pytest.mark.parametrize("entry", ["generator", "clones"])
def test_empty_public_task_entrypoints(entry: str) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("empty input must not create a Task")

    class EmptyTask:
        name = "empty"
        clone = forbidden

    options = dict(sequential=False, max_concurrency=2)
    result = (
        batch.run_batch_task_gen(forbidden, [], **options)
        if entry == "generator"
        else batch.run_batch_tasks(EmptyTask(), [], **options)
    )
    assert result == []
