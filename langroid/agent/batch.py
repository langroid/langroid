import asyncio
import copy
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    List,
    Optional,
    TypeVar,
    cast,
)

from dotenv import load_dotenv

from langroid.agent.base import Agent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.parsing.utils import batched
from langroid.utils.configuration import quiet_mode
from langroid.utils.logging import setup_colored_logging
from langroid.utils.output import SuppressLoggerWarnings, status

setup_colored_logging()

load_dotenv()

T = TypeVar("T")
U = TypeVar("U")

AsyncFunc = Callable[[T], Awaitable[U]]
SyncFunc = Callable[[T], U]


def run_batch_task_gen(
    gen_task: Callable[[int], Task],
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], U] = lambda x: x,  # type: ignore
    stop_on_first_result: bool = False,
    sequential: bool = True,
    batch_size: Optional[int] = None,
    turns: int = -1,
    message: Optional[str] = None,
    handle_exceptions: bool = False,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> list[Optional[U]]:
    """
    Generate and run copies of a task async/concurrently one per item in `items` list.
    For each item, apply `input_map` to get the initial message to process.
    For each result, apply `output_map` to get the final result.
    Args:
        gen_task (Callable[[int], Task]): generates the tasks to run
        items (list[T]): list of items to process
        input_map (Callable[[T], str|ChatDocument]): function to map item to
            initial message to process
        output_map (Callable[[ChatDocument|str], U]): function to map result
            to final result. If stop_on_first_result is enabled, then
            map any invalid output to None. We continue until some non-None
            result is obtained.
        stop_on_first_result (bool): whether to stop after the first valid
            (not-None) result. In this case all other tasks are
            cancelled, and their corresponding result is None in the
            returned list.
        sequential (bool): whether to run sequentially
            (e.g. some APIs such as ooba don't support concurrent requests)
        batch_size (Optional[int]): The number of tasks to run at a time,
            if None, unbatched
        turns (int): number of turns to run, -1 for infinite
        message (Optional[str]): optionally overrides the console status messages
        handle_exceptions: bool: Whether to replace exceptions with outputs of None
        max_cost: float: maximum cost to run the task (default 0.0 for unlimited)
        max_tokens: int: maximum token usage (in and out) (default 0 for unlimited)


    Returns:
        list[Optional[U]]: list of final results. Always list[U] if
        `stop_on_first_result` is disabled
    """
    inputs = [input_map(item) for item in items]

    async def _do_task(
        input: str | ChatDocument,
        i: int,
        return_idx: Optional[int] = None,
    ) -> BaseException | Optional[ChatDocument] | tuple[int, Optional[ChatDocument]]:
        task_i = gen_task(i)
        if task_i.agent.llm is not None:
            task_i.agent.llm.set_stream(False)
        task_i.agent.config.show_stats = False
        try:
            result = await task_i.run_async(
                input, turns=turns, max_cost=max_cost, max_tokens=max_tokens
            )
            if return_idx is not None:
                return return_idx, result
            else:
                return result
        except asyncio.CancelledError as e:
            task_i.kill()
            if handle_exceptions:
                return e
            else:
                raise e
        except BaseException as e:
            if handle_exceptions:
                return e
            else:
                raise e

    async def _do_all(
        inputs: Iterable[str | ChatDocument], start_idx: int = 0
    ) -> list[Optional[U]]:
        results: list[Optional[ChatDocument]] = []
        if stop_on_first_result:
            outputs: list[Optional[U]] = [None] * len(list(inputs))
            tasks = set(
                asyncio.create_task(_do_task(input, i + start_idx, return_idx=i))
                for i, input in enumerate(inputs)
            )
            while tasks:
                try:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        idx_result = task.result()
                        if not isinstance(idx_result, tuple):
                            continue
                        index, output = idx_result
                        outputs[index] = output_map(output)

                    if any(r is not None for r in outputs):
                        return outputs
                finally:
                    # Cancel all remaining tasks
                    for task in tasks:
                        task.cancel()
                    # Wait for cancellations to complete
                    try:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    except BaseException as e:
                        if not handle_exceptions:
                            raise e
            return outputs
        elif sequential:
            for i, input in enumerate(inputs):
                result: Optional[ChatDocument] | BaseException = await _do_task(
                    input, i + start_idx
                )  # type: ignore

                if isinstance(result, BaseException):
                    result = None

                results.append(result)
        else:
            results_with_exceptions = cast(
                list[Optional[ChatDocument | BaseException]],
                await asyncio.gather(
                    *(_do_task(input, i + start_idx) for i, input in enumerate(inputs)),
                ),
            )

            results = [
                r if not isinstance(r, BaseException) else None
                for r in results_with_exceptions
            ]

        return list(map(output_map, results))

    results: List[Optional[U]] = []
    if batch_size is None:
        msg = message or f"[bold green]Running {len(items)} tasks:"

        with status(msg), SuppressLoggerWarnings():
            results = asyncio.run(_do_all(inputs))
    else:
        batches = batched(inputs, batch_size)

        for batch in batches:
            start_idx = len(results)
            complete_str = f", {start_idx} complete" if start_idx > 0 else ""
            msg = message or f"[bold green]Running {len(items)} tasks{complete_str}:"

            if stop_on_first_result and any(r is not None for r in results):
                results.extend([None] * len(batch))
            else:
                with status(msg), SuppressLoggerWarnings():
                    results.extend(asyncio.run(_do_all(batch, start_idx=start_idx)))

    return results


def run_batch_tasks(
    task: Task,
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], U] = lambda x: x,  # type: ignore
    stop_on_first_result: bool = False,
    sequential: bool = True,
    batch_size: Optional[int] = None,
    turns: int = -1,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> List[Optional[U]]:
    """
    Run copies of `task` async/concurrently one per item in `items` list.
    For each item, apply `input_map` to get the initial message to process.
    For each result, apply `output_map` to get the final result.
    Args:
        task (Task): task to run
        items (list[T]): list of items to process
        input_map (Callable[[T], str|ChatDocument]): function to map item to
            initial message to process
        output_map (Callable[[ChatDocument|str], U]): function to map result
            to final result
        sequential (bool): whether to run sequentially
            (e.g. some APIs such as ooba don't support concurrent requests)
        batch_size (Optional[int]): The number of tasks to run at a time,
            if None, unbatched
        turns (int): number of turns to run, -1 for infinite
        max_cost: float: maximum cost to run the task (default 0.0 for unlimited)
        max_tokens: int: maximum token usage (in and out) (default 0 for unlimited)

    Returns:
        list[Optional[U]]: list of final results. Always list[U] if
        `stop_on_first_result` is disabled
    """
    message = f"[bold green]Running {len(items)} copies of {task.name}..."
    return run_batch_task_gen(
        lambda i: task.clone(i),
        items,
        input_map,
        output_map,
        stop_on_first_result,
        sequential,
        batch_size,
        turns,
        message,
        max_cost=max_cost,
        max_tokens=max_tokens,
    )


def run_batch_agent_method(
    agent: Agent,
    method: Callable[
        [str | ChatDocument | None], Coroutine[Any, Any, ChatDocument | None]
    ],
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
    sequential: bool = True,
    stop_on_first_result: bool = False,
    batch_size: Optional[int] = None,  # Added parameter
) -> List[Any]:
    """
    Run the `method` on copies of `agent`, async/concurrently one per
    item in `items` list.
    ASSUMPTION: The `method` is an async method and has signature:
        method(self, input: str|ChatDocument|None) -> ChatDocument|None
    So this would typically be used for the agent's "responder" methods,
    e.g. `llm_response_async` or `agent_responder_async`.

    For each item, apply `input_map` to get the initial message to process.
    For each result, apply `output_map` to get the final result.

    Args:
        agent (Agent): agent whose method to run
        method (str): Async method to run on copies of `agent`.
            The method is assumed to have signature:
            `method(self, input: str|ChatDocument|None) -> ChatDocument|None`
        input_map (Callable[[Any], str|ChatDocument]): function to map item to
            initial message to process
        output_map (Callable[[ChatDocument|str], Any]): function to map result
            to final result
        sequential (bool): whether to run sequentially
            (e.g. some APIs such as ooba don't support concurrent requests)
        batch_size (Optional[int]): The number of items to process in each batch.
            If None, process all items at once.
    Returns:
        List[Any]: list of final results
    """
    # Check if the method is async
    method_name = method.__name__
    if not inspect.iscoroutinefunction(method):
        raise ValueError(f"The method {method_name} is not async.")

    inputs = [input_map(item) for item in items]
    agent_cfg = copy.deepcopy(agent.config)
    assert agent_cfg.llm is not None, "agent must have llm config"
    agent_cfg.llm.stream = False
    agent_cfg.show_stats = False
    agent_cls = type(agent)
    agent_name = agent_cfg.name

    first_result = None
    first_result_lock = asyncio.Lock()

    async def _do_task(input_with_index: tuple[str | ChatDocument, int]) -> Any:
        """Process a single task with its input and index."""
        nonlocal first_result

        input_item, original_index = input_with_index
        agent_cfg.name = f"{agent_cfg.name}-{original_index}"
        agent_i = agent_cls(agent_cfg)
        method_i = getattr(agent_i, method_name, None)

        if method_i is None:
            raise ValueError(f"Agent {agent_name} has no method {method_name}")

        if stop_on_first_result:
            async with first_result_lock:
                if first_result is not None:
                    return None

                result = await method_i(input_item)
                processed_result = output_map(result)

                if processed_result is not None:
                    # Store both result and original index
                    first_result = (processed_result, original_index)
                return processed_result
        else:
            with quiet_mode(), SuppressLoggerWarnings():
                result = await method_i(input_item)
                return output_map(result)

    # Important: Create list of (input, index) pairs maintaining original order
    # This ensures task 0 processes first as expected by the test
    indexed_inputs = [(input_item, i) for i, input_item in enumerate(inputs)]

    with status(f"[bold green]Running {len(items)} copies of {agent_name}..."):
        # Use the batch utility function with original ordering
        results = run_batch_function(
            function=_do_task,
            items=indexed_inputs,
            sequential=sequential
            or stop_on_first_result,  # Force sequential if stop_on_first_result
            batch_size=(
                1 if stop_on_first_result else batch_size
            ),  # Force batch_size=1 if stopping on first
        )

        if stop_on_first_result:
            # Create a list of None values
            final_results = [None] * len(items)
            # If we got a result, place it at its original position
            if first_result is not None:
                result, original_index = first_result
                final_results[original_index] = result
            return final_results

        return results


def llm_response_batch(
    agent: Agent,
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
    sequential: bool = True,
    stop_on_first_result: bool = False,
    batch_size: Optional[int] = None,
) -> List[Any]:
    return run_batch_agent_method(
        agent,
        agent.llm_response_async,
        items,
        input_map=input_map,
        output_map=output_map,
        sequential=sequential,
        stop_on_first_result=stop_on_first_result,
        batch_size=batch_size,
    )


def agent_response_batch(
    agent: Agent,
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
    sequential: bool = True,
    stop_on_first_result: bool = False,
    batch_size: Optional[int] = None,
) -> List[Any]:
    return run_batch_agent_method(
        agent,
        agent.agent_response_async,
        items,
        input_map=input_map,
        output_map=output_map,
        sequential=sequential,
        stop_on_first_result=stop_on_first_result,
        batch_size=batch_size,
    )


async def run_all_batches(
    function: Callable[[T], U],
    items: List[T],
    sequential: bool,
    batch_size: Optional[int],
) -> List[U]:
    """Run a function on a list of items in batches.

    Args:
        function (Callable[[T], U]): Function to run on each item (sync or async)
        items (List[T]): List of items to process
        sequential (bool): If True, run tasks sequentially
        batch_size (Optional[int]): Number of items to process in each batch

    Returns:
        List[U]: List of results
    """
    is_async = inspect.iscoroutinefunction(function)

    async def _do_task(item: T) -> U:
        if is_async:
            # We know it's an async function in this branch
            async_fn = cast(AsyncFunc[T, U], function)
            return await async_fn(item)

        # We know it's a sync function in this branch
        sync_fn = function  # type: SyncFunc[T, U]
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_fn, item)
        return result

    async def _do_all(batch: Iterable[T]) -> List[U]:
        if sequential:
            results = []
            for item in batch:
                result = await _do_task(item)
                results.append(result)
            return results
        return await asyncio.gather(*(_do_task(item) for item in batch))

    results: List[U] = []

    if batch_size is None:
        # Single batch with all items
        batch_results = await _do_all(items)
        results.extend(batch_results)
    else:
        # Process in multiple batches
        batches = list(batched(items, batch_size))
        total_batches = len(batches)
        for i, batch in enumerate(batches, 1):
            with status(
                f"[bold green]Running batch {i}/{total_batches} of "
                f"{len(batch)} {'async ' if is_async else ''}tasks:"
            ):
                batch_results = await _do_all(batch)
                results.extend(batch_results)

    return results


def run_batch_function(
    function: Callable[[T], U],
    items: List[T],
    sequential: bool = True,
    batch_size: Optional[int] = None,
) -> List[U]:
    """Handle running a function on a list of items in batches.

    Function can be either sync or async.


    Args:
        function (Callable[[T], U]): Function to run on each item
        items (List[T]): List of items to process
        sequential (bool, optional): Defaults to True.
        batch_size (Optional[int], optional): Defaults to None.

    Returns:
        List[U]: List of results
    """
    return asyncio.run(run_all_batches(function, items, sequential, batch_size))
