"""Batch cancellation must propagate independently of error policy."""

import asyncio

import pytest

from langroid.agent.batch import (
    ExceptionHandling,
    _process_batch_async,
    run_batched_tasks,
)
from langroid.agent.chat_document import ChatDocument


@pytest.mark.parametrize("policy", list(ExceptionHandling))
def test_public_batch_propagates_cancellation(
    policy: ExceptionHandling,
) -> None:
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
