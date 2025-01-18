import asyncio
import copy
import inspect
import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
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


class ExceptionHandling(str, Enum):
    """Enum for exception handling options."""

    RAISE = "raise"
    RETURN_NONE = "return_none"
    RETURN_EXCEPTION = "return_exception"


def _convert_exception_handling(
    handle_exceptions: Union[bool, ExceptionHandling]
) -> ExceptionHandling:
    """Convert legacy boolean handle_exceptions to ExceptionHandling enum."""
    if isinstance(handle_exceptions, ExceptionHandling):
        return handle_exceptions

    if isinstance(handle_exceptions, bool):
        warnings.warn(
            "Boolean handle_exceptions is deprecated. "
            "Use ExceptionHandling enum instead: "
            "RAISE, RETURN_NONE, or RETURN_EXCEPTION.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (
            ExceptionHandling.RETURN_NONE
            if handle_exceptions
            else ExceptionHandling.RAISE
        )

    raise TypeError(
        "handle_exceptions must be bool or ExceptionHandling, "
        f"not {type(handle_exceptions)}"
    )


async def _process_batch_async(
    inputs: Iterable[str | ChatDocument],
    do_task: Callable[[str | ChatDocument, int], Coroutine[Any, Any, Any]],
    start_idx: int = 0,
    stop_on_first_result: bool = False,
    sequential: bool = False,
    handle_exceptions: Union[bool, ExceptionHandling] = ExceptionHandling.RAISE,
    output_map: Callable[[Any], Any] = lambda x: x,
) -> List[Optional[ChatDocument] | BaseException]:
    """
    Unified batch processing logic for both agent methods and tasks.

    Args:
        inputs: Iterable of inputs to process
        do_task: Task execution function that takes (input, index) and returns result
        start_idx: Starting index for the batch
        stop_on_first_result: Whether to stop after first valid result
        sequential: Whether to process sequentially
        handle_exceptions: How to handle exceptions:
            - RAISE or False: Let exceptions propagate
            - RETURN_NONE or True: Convert exceptions to None in results
            - RETURN_EXCEPTION: Include exception objects in results
            Boolean values are deprecated and will be removed in a future version.
        output_map: Function to map results to final output format
    """
    exception_handling = _convert_exception_handling(handle_exceptions)

    def handle_error(e: BaseException) -> Any:
        """Handle exceptions based on exception_handling."""
        match exception_handling:
            case ExceptionHandling.RAISE:
                raise e
            case ExceptionHandling.RETURN_NONE:
                return None
            case ExceptionHandling.RETURN_EXCEPTION:
                return e

    if stop_on_first_result:
        results: List[Optional[ChatDocument] | BaseException] = []
        pending: set[asyncio.Task[Any]] = set()
        # Create task-to-index mapping
        task_indices: dict[asyncio.Task[Any], int] = {}
        try:
            tasks = [
                asyncio.create_task(do_task(input, i + start_idx))
                for i, input in enumerate(inputs)
            ]
            task_indices = {task: i for i, task in enumerate(tasks)}
            results = [None] * len(tasks)

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                index = task_indices[task]
                try:
                    result = await task
                    results[index] = output_map(result)
                except BaseException as e:
                    results[index] = handle_error(e)

            if any(r is not None for r in results):
                return results
        finally:
            for task in pending:
                task.cancel()
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except BaseException as e:
                handle_error(e)
        return results

    elif sequential:
        results = []
        for i, input in enumerate(inputs):
            try:
                result = await do_task(input, i + start_idx)
                results.append(output_map(result))
            except BaseException as e:
                results.append(handle_error(e))
        return results

    # Parallel execution
    else:
        try:
            return_exceptions = exception_handling != ExceptionHandling.RAISE
            with quiet_mode(), SuppressLoggerWarnings():
                results_with_exceptions = cast(
                    list[Optional[ChatDocument | BaseException]],
                    await asyncio.gather(
                        *(
                            do_task(input, i + start_idx)
                            for i, input in enumerate(inputs)
                        ),
                        return_exceptions=return_exceptions,
                    ),
                )

                if exception_handling == ExceptionHandling.RETURN_NONE:
                    results = [
                        None if isinstance(r, BaseException) else r
                        for r in results_with_exceptions
                    ]
                else:  # ExceptionHandling.RETURN_EXCEPTION
                    results = results_with_exceptions
        except BaseException as e:
            results = [handle_error(e) for _ in inputs]

        return [output_map(r) for r in results]


def run_batched_tasks(
    inputs: List[str | ChatDocument],
    do_task: Callable[[str | ChatDocument, int], Coroutine[Any, Any, Any]],
    batch_size: Optional[int],
    stop_on_first_result: bool,
    sequential: bool,
    handle_exceptions: Union[bool, ExceptionHandling],
    output_map: Callable[[Any], Any],
    message_template: str,
    message: Optional[str] = None,
) -> List[Any]:
    """
    Common batch processing logic for both agent methods and tasks.

    Args:
        inputs: List of inputs to process
        do_task: Task execution function
        batch_size: Size of batches, if None process all at once
        stop_on_first_result: Whether to stop after first valid result
        sequential: Whether to process sequentially
        handle_exceptions: How to handle exceptions:
            - RAISE or False: Let exceptions propagate
            - RETURN_NONE or True: Convert exceptions to None in results
            - RETURN_EXCEPTION: Include exception objects in results
            Boolean values are deprecated and will be removed in a future version.
        output_map: Function to map results
        message_template: Template for status message
        message: Optional override for status message
    """

    async def run_all_batched_tasks(
        inputs: List[str | ChatDocument],
        batch_size: int | None,
    ) -> List[Any]:
        """Extra wrap to run asyncio.run one single time and not once per loop

        Args:
            inputs (List[str  |  ChatDocument]): inputs to process
            batch_size (int | None): batch size

        Returns:
            List[Any]: results
        """
        results: List[Any] = []
        if batch_size is None:
            msg = message or message_template.format(total=len(inputs))
            with status(msg), SuppressLoggerWarnings():
                results = await _process_batch_async(
                    inputs,
                    do_task,
                    stop_on_first_result=stop_on_first_result,
                    sequential=sequential,
                    handle_exceptions=handle_exceptions,
                    output_map=output_map,
                )
        else:
            batches = batched(inputs, batch_size)
            for batch in batches:
                start_idx = len(results)
                complete_str = f", {start_idx} complete" if start_idx > 0 else ""
                msg = (
                    message or message_template.format(total=len(inputs)) + complete_str
                )

                if stop_on_first_result and any(r is not None for r in results):
                    results.extend([None] * len(batch))
                else:
                    with status(msg), SuppressLoggerWarnings():
                        results.extend(
                            await _process_batch_async(
                                batch,
                                do_task,
                                start_idx=start_idx,
                                stop_on_first_result=stop_on_first_result,
                                sequential=sequential,
                                handle_exceptions=handle_exceptions,
                                output_map=output_map,
                            )
                        )
        return results

    return asyncio.run(run_all_batched_tasks(inputs, batch_size))


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
    handle_exceptions: Union[bool, ExceptionHandling] = ExceptionHandling.RAISE,
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
        handle_exceptions: How to handle exceptions:
            - RAISE or False: Let exceptions propagate
            - RETURN_NONE or True: Convert exceptions to None in results
            - RETURN_EXCEPTION: Include exception objects in results
            Boolean values are deprecated and will be removed in a future version.
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
    ) -> BaseException | Optional[ChatDocument] | tuple[int, Optional[ChatDocument]]:
        task_i = gen_task(i)
        if task_i.agent.llm is not None:
            task_i.agent.llm.set_stream(False)
        task_i.agent.config.show_stats = False

        try:
            result = await task_i.run_async(
                input, turns=turns, max_cost=max_cost, max_tokens=max_tokens
            )
        except asyncio.CancelledError as e:
            task_i.kill()
            # exception will be handled by the caller
            raise e
        return result

    return run_batched_tasks(
        inputs=inputs,
        do_task=_do_task,
        batch_size=batch_size,
        stop_on_first_result=stop_on_first_result,
        sequential=sequential,
        handle_exceptions=handle_exceptions,
        output_map=output_map,
        message_template="[bold green]Running {total} tasks:",
        message=message,
    )


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
    handle_exceptions: Union[bool, ExceptionHandling] = ExceptionHandling.RAISE,
    batch_size: Optional[int] = None,
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
        stop_on_first_result (bool): whether to stop after the first valid
        handle_exceptions: How to handle exceptions:
            - RAISE or False: Let exceptions propagate
            - RETURN_NONE or True: Convert exceptions to None in results
            - RETURN_EXCEPTION: Include exception objects in results
            Boolean values are deprecated and will be removed in a future version.
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

    async def _do_task(input: str | ChatDocument, i: int) -> Any:
        agent_cfg.name = f"{agent_cfg.name}-{i}"
        agent_i = agent_cls(agent_cfg)
        method_i = getattr(agent_i, method_name, None)
        if method_i is None:
            raise ValueError(f"Agent {agent_name} has no method {method_name}")
        result = await method_i(input)
        return result

    return run_batched_tasks(
        inputs=inputs,
        do_task=_do_task,
        batch_size=batch_size,
        stop_on_first_result=stop_on_first_result,
        sequential=sequential,
        handle_exceptions=handle_exceptions,
        output_map=output_map,
        message_template=f"[bold green]Running {{total}} copies of {agent_name}...",
    )


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


def run_batch_function(
    function: Callable[[T], U],
    items: list[T],
    sequential: bool = True,
    batch_size: Optional[int] = None,
) -> List[U]:
    async def _do_task(item: T) -> U:
        return function(item)

    async def _do_all(items: Iterable[T]) -> List[U]:
        if sequential:
            results = []
            for item in items:
                result = await _do_task(item)
                results.append(result)
            return results

        return await asyncio.gather(*(_do_task(item) for item in items))

    results: List[U] = []

    if batch_size is None:
        with status(f"[bold green]Running {len(items)} tasks:"):
            results = asyncio.run(_do_all(items))
    else:
        batches = batched(items, batch_size)
        for batch in batches:
            with status(f"[bold green]Running batch of {len(batch)} tasks:"):
                results.extend(asyncio.run(_do_all(batch)))

    return results
