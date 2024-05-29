from typing import Any

from _typeshed import Incomplete
from pydantic import BaseModel

from langroid.agent.base import Agent as Agent
from langroid.agent.chat_agent import ChatAgent as ChatAgent
from langroid.agent.chat_document import (
    ChatDocLoggerFields as ChatDocLoggerFields,
)
from langroid.agent.chat_document import (
    ChatDocMetaData as ChatDocMetaData,
)
from langroid.agent.chat_document import (
    ChatDocument as ChatDocument,
)
from langroid.agent.chat_document import (
    StatusCode as StatusCode,
)
from langroid.cachedb.redis_cachedb import (
    RedisCache as RedisCache,
)
from langroid.cachedb.redis_cachedb import (
    RedisCacheConfig as RedisCacheConfig,
)
from langroid.exceptions import InfiniteLoopException as InfiniteLoopException
from langroid.mytypes import Entity as Entity
from langroid.parsing.parse_json import extract_top_level_json as extract_top_level_json
from langroid.parsing.routing import parse_addressed_message as parse_addressed_message
from langroid.utils.configuration import settings as settings
from langroid.utils.constants import (
    DONE as DONE,
)
from langroid.utils.constants import (
    NO_ANSWER as NO_ANSWER,
)
from langroid.utils.constants import (
    PASS as PASS,
)
from langroid.utils.constants import (
    PASS_TO as PASS_TO,
)
from langroid.utils.constants import (
    SEND_TO as SEND_TO,
)
from langroid.utils.constants import (
    USER_QUIT_STRINGS as USER_QUIT_STRINGS,
)
from langroid.utils.logging import (
    RichFileLogger as RichFileLogger,
)
from langroid.utils.logging import (
    setup_file_logger as setup_file_logger,
)
from langroid.utils.system import hash as hash

logger: Incomplete
Responder: Incomplete

def noop_fn(*args: list[Any], **kwargs: dict[str, Any]) -> None: ...

class TaskConfig(BaseModel):
    inf_loop_cycle_len: int
    inf_loop_dominance_factor: float
    inf_loop_wait_factor: float

class Task:
    cache: RedisCache
    callbacks: Incomplete
    config: Incomplete
    message_counter: Incomplete
    history_count: Incomplete
    max_cost: int
    max_tokens: int
    session_id: str
    logger: Incomplete
    tsv_logger: Incomplete
    color_log: Incomplete
    agent: Incomplete
    step_progress: bool
    n_stalled_steps: int
    max_stalled_steps: Incomplete
    done_if_response: Incomplete
    done_if_no_response: Incomplete
    is_done: bool
    is_pass_thru: bool
    task_progress: bool
    name: Incomplete
    value: Incomplete
    interactive: Incomplete
    message_history_idx: int
    default_human_response: Incomplete
    only_user_quits_root: Incomplete
    erase_substeps: Incomplete
    allow_null_result: Incomplete
    responders: Incomplete
    responders_async: Incomplete
    non_human_responders: Incomplete
    non_human_responders_async: Incomplete
    human_tried: bool
    name_sub_task_map: Incomplete
    pending_message: Incomplete
    pending_sender: Incomplete
    single_round: Incomplete
    turns: int
    llm_delegate: Incomplete
    controller: Incomplete
    sub_tasks: Incomplete
    parent_task: Incomplete
    caller: Incomplete
    def __init__(
        self,
        agent: Agent | None = None,
        name: str = "",
        llm_delegate: bool = False,
        single_round: bool = False,
        system_message: str = "",
        user_message: str | None = "",
        restart: bool = True,
        default_human_response: str | None = None,
        interactive: bool = True,
        only_user_quits_root: bool = False,
        erase_substeps: bool = False,
        allow_null_result: bool = True,
        max_stalled_steps: int = 5,
        done_if_no_response: list[Responder] = [],
        done_if_response: list[Responder] = [],
        config: TaskConfig = ...,
    ) -> None: ...
    def clone(self, i: int) -> Task: ...
    @classmethod
    def kill_session(cls, session_id: str = "") -> None: ...
    def kill(self) -> None: ...
    def add_sub_task(self, task: Task | list[Task]) -> None: ...
    def init(self, msg: None | str | ChatDocument = None) -> ChatDocument | None: ...
    def run(
        self,
        msg: str | ChatDocument | None = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
    ) -> ChatDocument | None: ...
    async def run_async(
        self,
        msg: str | ChatDocument | None = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
    ) -> ChatDocument | None: ...
    def step(self, turns: int = -1) -> ChatDocument | None: ...
    async def step_async(self, turns: int = -1) -> ChatDocument | None: ...
    def response(self, e: Responder, turns: int = -1) -> ChatDocument | None: ...
    async def response_async(
        self, e: Responder, turns: int = -1
    ) -> ChatDocument | None: ...
    def result(self) -> ChatDocument: ...
    def done(
        self, result: ChatDocument | None = None, r: Responder | None = None
    ) -> tuple[bool, StatusCode]: ...
    def valid(self, result: ChatDocument | None, r: Responder) -> bool: ...
    def log_message(
        self, resp: Responder, msg: ChatDocument | None = None, mark: bool = False
    ) -> None: ...
    def set_color_log(self, enable: bool = True) -> None: ...

def parse_routing(
    msg: ChatDocument | str,
) -> tuple[bool | None, str | None, str | None]: ...
