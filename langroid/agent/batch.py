import asyncio
import copy
import inspect
from typing import Any, Callable, Coroutine, Iterable, List, Optional, TypeVar

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


def run_batch_task_gen(
    gen_task: Callable[[int], Task],
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], U] = lambda x: x,  # type: ignore
    sequential: bool = True,
    batch_size: Optional[int] = None,
    turns: int = -1,
    message: Optional[str] = None,
    handle_exceptions: bool = False,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> list[U]:
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
            to final result
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
        list[Any]: list of final results
    """
    inputs = [input_map(item) for item in items]

    async def _do_task(input: str | ChatDocument, i: int) -> Optional[ChatDocument]:
        task_i = gen_task(i)
        if task_i.agent.llm is not None:
            task_i.agent.llm.set_stream(False)
        task_i.agent.config.show_stats = False

        result = await task_i.run_async(
            input, turns=turns, max_cost=max_cost, max_tokens=max_tokens
        )
        return result

    async def _do_all(
        inputs: Iterable[str | ChatDocument], start_idx: int = 0
    ) -> list[U]:
        results: list[Optional[ChatDocument]] = []
        if sequential:
            for i, input in enumerate(inputs):
                try:
                    result = await _do_task(input, i + start_idx)
                except BaseException as e:
                    if handle_exceptions:
                        result = None
                    else:
                        raise e
                results.append(result)
        else:
            results_with_exceptions = await asyncio.gather(
                *(_do_task(input, i + start_idx) for i, input in enumerate(inputs)),
                return_exceptions=handle_exceptions,
            )

            results = [
                r if not isinstance(r, BaseException) else None
                for r in results_with_exceptions
            ]

        return list(map(output_map, results))

    results: List[U] = []
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

            with status(msg), SuppressLoggerWarnings():
                results.extend(asyncio.run(_do_all(batch, start_idx=start_idx)))

    return results


def run_batch_tasks(
    task: Task,
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], U] = lambda x: x,  # type: ignore
    sequential: bool = True,
    batch_size: Optional[int] = None,
    turns: int = -1,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> List[U]:
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
        list[Any]: list of final results
    """
    message = f"[bold green]Running {len(items)} copies of {task.name}..."
    return run_batch_task_gen(
        lambda i: task.clone(i),
        items,
        input_map,
        output_map,
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
        return output_map(result)

    async def _do_all() -> List[Any]:
        if sequential:
            results = []
            for i, input in enumerate(inputs):
                result = await _do_task(input, i)
                results.append(result)
            return results
        with quiet_mode(), SuppressLoggerWarnings():
            return await asyncio.gather(
                *(_do_task(input, i) for i, input in enumerate(inputs))
            )

    n = len(items)
    with status(f"[bold green]Running {n} copies of {agent_name}..."):
        results = asyncio.run(_do_all())

    return results


def llm_response_batch(
    agent: Agent,
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
    sequential: bool = True,
) -> List[Any]:
    return run_batch_agent_method(
        agent,
        agent.llm_response_async,
        items,
        input_map=input_map,
        output_map=output_map,
        sequential=sequential,
    )


def agent_response_batch(
    agent: Agent,
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
    sequential: bool = True,
) -> List[Any]:
    return run_batch_agent_method(
        agent,
        agent.agent_response_async,
        items,
        input_map=input_map,
        output_map=output_map,
        sequential=sequential,
    )
