from typing import Any, Callable, Coroutine, TypeVar

from langroid.agent.base import Agent as Agent
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.task import Task as Task
from langroid.parsing.utils import batched as batched
from langroid.utils.configuration import quiet_mode as quiet_mode
from langroid.utils.logging import setup_colored_logging as setup_colored_logging
from langroid.utils.output import (
    SuppressLoggerWarnings as SuppressLoggerWarnings,
)
from langroid.utils.output import (
    status as status,
)

T = TypeVar("T")
U = TypeVar("U")

def run_batch_task_gen(
    gen_task: Callable[[int], Task],
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = ...,
    output_map: Callable[[ChatDocument | None], U] = ...,
    sequential: bool = True,
    batch_size: int | None = None,
    turns: int = -1,
    message: str | None = None,
    handle_exceptions: bool = False,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> list[U]: ...
def run_batch_tasks(
    task: Task,
    items: list[T],
    input_map: Callable[[T], str | ChatDocument] = ...,
    output_map: Callable[[ChatDocument | None], U] = ...,
    sequential: bool = True,
    batch_size: int | None = None,
    turns: int = -1,
    max_cost: float = 0.0,
    max_tokens: int = 0,
) -> list[U]: ...
def run_batch_agent_method(
    agent: Agent,
    method: Callable[
        [str | ChatDocument | None], Coroutine[Any, Any, ChatDocument | None]
    ],
    items: list[Any],
    input_map: Callable[[Any], str | ChatDocument] = ...,
    output_map: Callable[[ChatDocument | None], Any] = ...,
    sequential: bool = True,
) -> list[Any]: ...
def llm_response_batch(
    agent: Agent,
    items: list[Any],
    input_map: Callable[[Any], str | ChatDocument] = ...,
    output_map: Callable[[ChatDocument | None], Any] = ...,
    sequential: bool = True,
) -> list[Any]: ...
def agent_response_batch(
    agent: Agent,
    items: list[Any],
    input_map: Callable[[Any], str | ChatDocument] = ...,
    output_map: Callable[[ChatDocument | None], Any] = ...,
    sequential: bool = True,
) -> list[Any]: ...
