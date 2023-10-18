import asyncio
from typing import Any, Callable, List

from dotenv import load_dotenv
from rich.console import Console

from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.utils.logging import setup_colored_logging

console = Console()

setup_colored_logging()

load_dotenv()


def run_batch_tasks(
    task: Task,
    items: List[Any],
    input_map: Callable[[Any], str | ChatDocument] = lambda x: str(x),
    output_map: Callable[[ChatDocument | None], Any] = lambda x: x,
) -> List[Any]:
    """
    Run copies of `task` async/concurrently one per item in `items` list.
    For each item, apply `input_map` to get the initial message to process.
    For each result, apply `output_map` to get the final result.
    Args:
        items (List[Any]): list of items to process
        turns (int): number of turns to run the task for;
            default is -1, which means run until task is done.
        input_map (Callable[[Any], str|ChatDocument]): function to map item to
            initial message to process
        output_map (Callable[[ChatDocument|str], Any]): function to map result
            to final result

    Returns:
        List[Any]: list of final results
    """

    inputs = [input_map(item) for item in items]

    async def _do_task(input: str | ChatDocument, i: int) -> Any:
        task_i = task.clone(i)
        result = await task_i.run_async(input)
        return output_map(result)

    async def _do_all() -> List[Any]:
        return await asyncio.gather(  # type: ignore
            *(_do_task(input, i) for i, input in enumerate(inputs))
        )

    # show rich console spinner

    n = len(items)
    with console.status(f"[bold green]Running {n} copies of {task.name}..."):
        results = asyncio.run(_do_all())

    return results
