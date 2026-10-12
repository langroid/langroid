"""Batch cancellation must propagate independently of error policy."""

import asyncio
import sys

import pytest

from langroid.agent.batch import (
    ExceptionHandling,
    _process_batch_async,
    run_batched_tasks,
)
from langroid.agent.chat_document import ChatDocument


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "policy",
    [ExceptionHandling.RETURN_NONE, ExceptionHandling.RETURN_EXCEPTION],
)
def test_sequential_batch_propagates_process_exit(
    error: type[BaseException], policy: ExceptionHandling
) -> None:
    started: list[int] = []

    async def work(value: str | ChatDocument, index: int) -> str:
        started.append(index)
        raise error("stop batch")

    with pytest.raises(error, match="stop batch"):
        run_batched_tasks(
            inputs=["a", "b"],
            do_task=work,
            batch_size=None,
            stop_on_first_result=False,
            sequential=True,
            handle_exceptions=policy,
            output_map=lambda value: value,
            message_template="Testing process exit",
        )

    assert started == [0]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "policy",
    [ExceptionHandling.RETURN_NONE, ExceptionHandling.RETURN_EXCEPTION],
)
@pytest.mark.parametrize("mode", ["sequential", "parallel", "first_result"])
def test_batch_propagates_process_exit_from_output_map(
    error: type[BaseException], policy: ExceptionHandling, mode: str
) -> None:
    mapped: list[str] = []
    cleaned: list[int] = []
    release = asyncio.Event()

    async def work(value: str | ChatDocument, index: int) -> str:
        try:
            if mode == "first_result" and index == 1:
                await release.wait()
            return str(value)
        finally:
            cleaned.append(index)

    def output_map(value: str) -> str:
        mapped.append(value)
        release.set()
        raise error("stop mapping")

    with pytest.raises(error, match="stop mapping"):
        run_batched_tasks(
            inputs=["a", "b"],
            do_task=work,
            batch_size=None,
            stop_on_first_result=mode == "first_result",
            sequential=mode == "sequential",
            handle_exceptions=policy,
            output_map=output_map,
            message_template="Testing process exit",
        )

    assert mapped == ["a"]
    assert sorted(cleaned) == ([0] if mode == "sequential" else [0, 1])


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_first_result_process_exit_survives_cleanup_cancellation(
    error: type[BaseException], policy: ExceptionHandling
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleaning = asyncio.Event()
        cleaned = asyncio.Event()
        exit_error = error("stop mapping")

        async def work(value: str | ChatDocument, index: int) -> str:
            if index == 0:
                await started.wait()
                return "winner"
            try:
                started.set()
                await asyncio.Event().wait()
                return "unused"
            finally:
                cleaning.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

        def output_map(value: str) -> str:
            raise exit_error

        async def capture_batch_exit() -> BaseException:
            # Catch process exits inside the task so asyncio's runner stays alive.
            try:
                await _process_batch_async(
                    ["a", "b"],
                    work,
                    stop_on_first_result=True,
                    handle_exceptions=policy,
                    output_map=output_map,
                )
            except BaseException as actual:
                return actual
            raise AssertionError("The batch should propagate an exception")

        batch = asyncio.create_task(capture_batch_exit())
        try:
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            batch.cancel()
            actual = await asyncio.wait_for(batch, timeout=2)
            assert actual is exit_error
            assert cleaned.is_set()
        finally:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)

    asyncio.run(scenario())


def _require_cancellation_detection(policy: ExceptionHandling) -> None:
    """Skip where a policy-handled CancelledError cannot be told from a
    cancellation of the batch: that needs `Task.cancelling()` (Python 3.11+).
    See docs/notes/batch-processing.md."""
    if policy != ExceptionHandling.RAISE and sys.version_info < (3, 11):
        pytest.skip("external batch cancellation is only detected on Python 3.11+")


@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_public_batch_propagates_cancellation(
    policy: ExceptionHandling,
) -> None:
    _require_cancellation_detection(policy)

    async def work(value: str | ChatDocument, index: int) -> str:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        await asyncio.sleep(0)
        return str(value)

    with pytest.raises(asyncio.CancelledError):
        run_batched_tasks(
            inputs=["a"],
            do_task=work,
            batch_size=None,
            stop_on_first_result=False,
            sequential=True,
            handle_exceptions=policy,
            output_map=lambda value: value,
            message_template="Testing cancellation",
        )


@pytest.mark.parametrize("policy", list(ExceptionHandling))
@pytest.mark.parametrize("mode", ["sequential", "parallel", "first_result"])
@pytest.mark.parametrize("iterable", [False, True])
def test_batch_propagates_cancellation(
    policy: ExceptionHandling, mode: str, iterable: bool
) -> None:
    if mode == "sequential":
        _require_cancellation_detection(policy)

    async def scenario() -> None:
        started: list[int] = []
        cleaned: list[int] = []
        ready = asyncio.Event()
        expected = 1 if mode == "sequential" else 2

        async def work(value: str | ChatDocument, index: int) -> str:
            started.append(index)
            if len(started) == expected:
                ready.set()
            try:
                if mode == "sequential" and index > 0:
                    return str(value)
                await asyncio.Event().wait()
                return str(value)
            finally:
                cleaned.append(index)

        inputs = ["a", "b"]
        batch = asyncio.create_task(
            _process_batch_async(
                iter(inputs) if iterable else inputs,
                work,
                sequential=mode == "sequential",
                stop_on_first_result=mode == "first_result",
                handle_exceptions=policy,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(batch, timeout=2)
            assert sorted(started) == list(range(expected))
            assert sorted(cleaned) == list(range(expected))
        finally:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_batch_first_result_still_cleans_pending(
    policy: ExceptionHandling,
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def work(value: str | ChatDocument, index: int) -> str:
            if index == 0:
                await started.wait()
                return "winner"
            try:
                started.set()
                await asyncio.Event().wait()
                return "unused"
            finally:
                cleaned.set()

        result = await asyncio.wait_for(
            _process_batch_async(
                ["a", "b"],
                work,
                stop_on_first_result=True,
                handle_exceptions=policy,
            ),
            timeout=2,
        )
        assert result == ["winner", None]
        assert cleaned.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_batch_cancellation_during_first_result_cleanup(
    policy: ExceptionHandling,
) -> None:
    _require_cancellation_detection(policy)

    async def scenario() -> None:
        started = asyncio.Event()
        cleaning = asyncio.Event()
        cleaned = asyncio.Event()

        async def work(value: str | ChatDocument, index: int) -> str:
            if index == 0:
                await started.wait()
                return "winner"
            try:
                started.set()
                await asyncio.Event().wait()
                return "unused"
            finally:
                cleaning.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

        batch = asyncio.create_task(
            _process_batch_async(
                ["a", "b"],
                work,
                stop_on_first_result=True,
                handle_exceptions=policy,
            )
        )
        try:
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(batch, timeout=2)
            assert cleaned.is_set()
        finally:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(ExceptionHandling))
@pytest.mark.parametrize("mode", ["sequential", "parallel", "first_result"])
def test_task_raised_cancelled_error_follows_policy(
    policy: ExceptionHandling, mode: str
) -> None:
    """A CancelledError raised by a task's own code is a task failure."""

    async def work(value: str | ChatDocument, index: int) -> str:
        if index == 1:
            raise asyncio.CancelledError()
        return str(value)

    async def scenario() -> list[object]:
        return await _process_batch_async(
            ["a", "b", "c"],
            work,
            sequential=mode == "sequential",
            stop_on_first_result=mode == "first_result",
            handle_exceptions=policy,
        )

    if policy == ExceptionHandling.RAISE:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(scenario())
        return

    results = asyncio.run(scenario())
    assert len(results) == 3
    if mode == "first_result":
        # One valid result stops the batch; the other slots are None.
        assert "a" in results or "c" in results
        assert results[1] is None or isinstance(results[1], asyncio.CancelledError)
        return
    assert results[0] == "a" and results[2] == "c"
    if policy == ExceptionHandling.RETURN_NONE:
        assert results[1] is None
    else:
        assert isinstance(results[1], asyncio.CancelledError)


@pytest.mark.parametrize("iterable", [False, True])
def test_parallel_raise_policy_with_iterator_input(iterable: bool) -> None:
    """Under RAISE, a task failure propagates even when inputs is an iterator."""

    async def work(value: str | ChatDocument, index: int) -> str:
        if index == 1:
            raise ValueError("task failed")
        return str(value)

    async def scenario() -> list[object]:
        inputs = ["a", "b"]
        return await _process_batch_async(
            iter(inputs) if iterable else inputs,
            work,
            handle_exceptions=ExceptionHandling.RAISE,
        )

    with pytest.raises(ValueError, match="task failed"):
        asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_parallel_maps_returned_exception_object(
    policy: ExceptionHandling,
) -> None:
    """An exception object *returned* by a task is a result under any policy."""

    async def work(value: str | ChatDocument, index: int) -> object:
        return ValueError(str(value))

    async def scenario() -> list[object]:
        return await _process_batch_async(
            ["a", "b"],
            work,
            handle_exceptions=policy,
            output_map=lambda r: f"mapped:{r}",
        )

    assert asyncio.run(scenario()) == ["mapped:a", "mapped:b"]
