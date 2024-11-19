from __future__ import annotations

import asyncio
import copy
import logging
import re
import threading
from collections import Counter, OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from rich import print
from rich.markup import escape

from langroid.agent.base import Agent
from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import (
    ChatDocLoggerFields,
    ChatDocMetaData,
    ChatDocument,
    StatusCode,
)
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool, DoneTool, FinalResultTool
from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.exceptions import InfiniteLoopException
from langroid.mytypes import Entity
from langroid.parsing.parse_json import extract_top_level_json
from langroid.parsing.routing import parse_addressed_message
from langroid.pydantic_v1 import BaseModel
from langroid.utils.configuration import settings
from langroid.utils.constants import (
    DONE,
    NO_ANSWER,
    PASS,
    PASS_TO,
    SEND_TO,
    USER_QUIT_STRINGS,
)
from langroid.utils.logging import RichFileLogger, setup_file_logger
from langroid.utils.object_registry import scheduled_cleanup
from langroid.utils.system import hash
from langroid.utils.types import to_string

logger = logging.getLogger(__name__)

Responder = Entity | Type["Task"]

T = TypeVar("T")


def noop_fn(*args: List[Any], **kwargs: Dict[str, Any]) -> None:
    pass


class TaskConfig(BaseModel):
    """Configuration for a Task. This is a container for any params that
    we didn't include in the task `__init__` method.
    We may eventually move all the task __init__ params to this class, analogous to how
    we have config classes for `Agent`, `ChatAgent`, `LanguageModel`, etc.

    Attributes:
        inf_loop_cycle_len (int): max exact-loop cycle length: 0 => no inf loop test
        inf_loop_dominance_factor (float): dominance factor for exact-loop detection
        inf_loop_wait_factor (int): wait this * cycle_len msgs before loop-check
        restart_as_subtask (bool): whether to restart *every* run of this task
            when run as a subtask.
        addressing_prefix (str): "@"-like prefix an agent can use to address other
            agents, or entities of the agent. E.g., if this is "@", the addressing
            string would be "@Alice", or "@user", "@llm", "@agent", etc.
            If this is an empty string, then addressing is disabled.
            Default is empty string "".
            CAUTION: this is a deprecated practice, since normal prompts
            can accidentally contain such addressing prefixes, and will break
            your runs. This could happen especially when your prompt/context
            contains code, but of course could occur in normal text as well.
            Instead, use the `RecipientTool` to have agents address other agents or
            entities. If you do choose to use `addressing_prefix`, the recommended
            setting is to use `langroid.utils.constants.AT`, which currently is "|@|".
            Note that this setting does NOT affect the use of `constants.SEND_TO` --
            this is always enabled since this is a critical way for responders to
            indicate that the message should be sent to a specific entity/agent.
            (Search for "SEND_TO" in the examples/ dir to see how this is used.)
        allow_subtask_multi_oai_tools (bool): whether to allow multiple OpenAI
            tool-calls to be sent to a sub-task.
        recognize_string_signals (bool): whether to recognize string-based signaling
            like DONE, SEND_TO, PASS, etc. Default is True, but note that we don't need
            to use string-based signaling, and it is recommended to use the
            new Orchestration tools instead (see agent/tools/orchestration.py),
            e.g. DoneTool, SendTool, etc.

    """

    inf_loop_cycle_len: int = 10
    inf_loop_dominance_factor: float = 1.5
    inf_loop_wait_factor: int = 5
    restart_as_subtask: bool = False
    logs_dir: str = "logs"
    addressing_prefix: str = ""
    allow_subtask_multi_oai_tools: bool = True
    recognize_string_signals: bool = True


class Task:
    """
    A `Task` wraps an `Agent` object, and sets up the `Agent`'s goals and instructions.
    A `Task` maintains two key variables:

    - `self.pending_message`, which is the message awaiting a response, and
    - `self.pending_sender`, which is the entity that sent the pending message.

    The possible responders to `self.pending_message` are the `Agent`'s own "native"
    responders (`agent_response`, `llm_response`, and `user_response`), and
    the `run()` methods of any sub-tasks. All responders have the same type-signature
    (somewhat simplified):
    ```
    str | ChatDocument -> ChatDocument
    ```
    Responders may or may not specify an intended recipient of their generated response.

    The main top-level method in the `Task` class is `run()`, which repeatedly calls
    `step()` until `done()` returns true. The `step()` represents a "turn" in the
    conversation: this method sequentially (in round-robin fashion) calls the responders
    until it finds one that generates a *valid* response to the `pending_message`
    (as determined by the `valid()` method). Once a valid response is found,
    `step()` updates the `pending_message` and `pending_sender` variables,
    and on the next iteration, `step()` re-starts its search for a valid response
    *from the beginning* of the list of responders (the exception being that the
    human user always gets a chance to respond after each non-human valid response).
    This process repeats until `done()` returns true, at which point `run()` returns
    the value of `result()`, which is the final result of the task.
    """

    # class variable called `cache` that is a RedisCache object
    _cache: RedisCache | None = None
    _background_tasks_started: bool = False

    def __init__(
        self,
        agent: Optional[Agent] = None,
        name: str = "",
        llm_delegate: bool = False,
        single_round: bool = False,
        system_message: str = "",
        user_message: str | None = "",
        restart: bool = True,
        default_human_response: Optional[str] = None,
        interactive: bool = True,
        only_user_quits_root: bool = True,
        erase_substeps: bool = False,
        allow_null_result: bool = False,
        max_stalled_steps: int = 5,
        default_return_type: Optional[type] = None,
        done_if_no_response: List[Responder] = [],
        done_if_response: List[Responder] = [],
        config: TaskConfig = TaskConfig(),
        **kwargs: Any,  # catch-all for any legacy params, for backwards compatibility
    ):
        """
        A task to be performed by an agent.

        Args:
            agent (Agent): agent associated with the task
            name (str): name of the task
            llm_delegate (bool):
                Whether to delegate "control" to LLM; conceptually,
                the "controlling entity" is the one "seeking" responses to its queries,
                and has a goal it is aiming to achieve, and decides when a task is done.
                The "controlling entity" is either the LLM or the USER.
                (Note within a Task there is just one
                LLM, and all other entities are proxies of the "User" entity).
                See also: `done_if_response`, `done_if_no_response` for more granular
                control of task termination.
            single_round (bool):
                If true, task runs until one message by "controller"
                (i.e. LLM if `llm_delegate` is true, otherwise USER)
                and subsequent response by non-controller [When a tool is involved,
                this will not give intended results. See `done_if_response`,
                `done_if_no_response` below].
                termination]. If false, runs for the specified number of turns in
                `run`, or until `done()` is true.
                One run of step() is considered a "turn".
                See also: `done_if_response`, `done_if_no_response` for more granular
                control of task termination.
            system_message (str): if not empty, overrides agent's system_message
            user_message (str): if not empty, overrides agent's user_message
            restart (bool): if true, resets the agent's message history *at every run*.
            default_human_response (str|None): default response from user; useful for
                testing, to avoid interactive input from user.
                [Instead of this, setting `interactive` usually suffices]
            default_return_type: if not None, extracts a value of this type from the
                result of self.run()
            interactive (bool): if true, wait for human input after each non-human
                response (prevents infinite loop of non-human responses).
                Default is true. If false, then `default_human_response` is set to ""
                Note: When interactive = False, the one exception is when the user
                is explicitly addressed, via "@user" or using RecipientTool, in which
                case the system will wait for a user response. In other words, use
                `interactive=False` when you want a "largely non-interactive"
                run, with the exception of explicit user addressing.
            only_user_quits_root (bool): if true, when interactive=True, only user can
                quit the root task (Ignored when interactive=False).
            erase_substeps (bool): if true, when task completes, erase intermediate
                conversation with subtasks from this agent's `message_history`, and also
                erase all subtask agents' `message_history`.
                Note: erasing can reduce prompt sizes, but results in repetitive
                sub-task delegation.
            allow_null_result (bool):
                If true, create dummy NO_ANSWER response when no valid response is found
                in a step.
                Optional, default is False.
                *Note:* In non-interactive mode, when this is set to True,
                you can have a situation where an LLM generates (non-tool) text,
                and no other responders have valid responses, and a "Null result"
                is inserted as a dummy response from the User entity, so the LLM
                will now respond to this Null result, and this will continue
                until the LLM emits a DONE signal (if instructed to do so),
                otherwise langroid detects a potential infinite loop after
                a certain number of such steps (= `TaskConfig.inf_loop_wait_factor`)
                and will raise an InfiniteLoopException.
            max_stalled_steps (int): task considered done after this many consecutive
                steps with no progress. Default is 3.
            done_if_no_response (List[Responder]): consider task done if NULL
                response from any of these responders. Default is empty list.
            done_if_response (List[Responder]): consider task done if NON-NULL
                response from any of these responders. Default is empty list.
        """
        if agent is None:
            agent = ChatAgent()
        self.callbacks = SimpleNamespace(
            show_subtask_response=noop_fn,
            set_parent_agent=noop_fn,
        )
        self.config = config
        # how to behave as a sub-task; can be overridden by `add_sub_task()`
        self.config_sub_task = copy.deepcopy(config)
        # counts of distinct pending messages in history,
        # to help detect (exact) infinite loops
        self.message_counter: Counter[str] = Counter()
        self._init_message_counter()

        self.history: Deque[str] = deque(
            maxlen=self.config.inf_loop_cycle_len * self.config.inf_loop_wait_factor
        )
        # copy the agent's config, so that we don't modify the original agent's config,
        # which may be shared by other agents.
        try:
            config_copy = copy.deepcopy(agent.config)
            agent.config = config_copy
        except Exception:
            logger.warning(
                """
                Failed to deep-copy Agent config during task creation, 
                proceeding with original config. Be aware that changes to 
                the config may affect other agents using the same config.
                """
            )
        self.restart = restart
        agent = cast(ChatAgent, agent)
        self.agent: ChatAgent = agent
        if isinstance(agent, ChatAgent) and len(agent.message_history) == 0 or restart:
            self.agent.init_state()
            # possibly change the system and user messages
            if system_message:
                # we always have at least 1 task_message
                self.agent.set_system_message(system_message)
            if user_message:
                self.agent.set_user_message(user_message)
        self.max_cost: float = 0
        self.max_tokens: int = 0
        self.session_id: str = ""
        self.logger: None | RichFileLogger = None
        self.tsv_logger: None | logging.Logger = None
        self.color_log: bool = False if settings.notebook else True

        self.n_stalled_steps = 0  # how many consecutive steps with no progress?
        # how many 2-step-apart alternations of no_answer step-result have we had,
        # i.e. x1, N/A, x2, N/A, x3, N/A ...
        self.n_no_answer_alternations = 0
        self._no_answer_step: int = -5
        self._step_idx = -1  # current step index
        self.max_stalled_steps = max_stalled_steps
        self.done_if_response = [r.value for r in done_if_response]
        self.done_if_no_response = [r.value for r in done_if_no_response]
        self.is_done = False  # is task done (based on response)?
        self.is_pass_thru = False  # is current response a pass-thru?
        if name:
            # task name overrides name in agent config
            agent.config.name = name
        self.name = name or agent.config.name
        self.value: str = self.name

        self.default_human_response = default_human_response
        if default_human_response is not None:
            # only override agent's default_human_response if it is explicitly set
            self.agent.default_human_response = default_human_response
        self.interactive = interactive
        self.agent.interactive = interactive
        self.only_user_quits_root = only_user_quits_root
        self.message_history_idx = -1
        self.default_return_type = default_return_type

        # set to True if we want to collapse multi-turn conversation with sub-tasks into
        # just the first outgoing message and last incoming message.
        # Note this also completely erases sub-task agents' message_history.
        self.erase_substeps = erase_substeps
        self.allow_null_result = allow_null_result

        agent_entity_responders = agent.entity_responders()
        agent_entity_responders_async = agent.entity_responders_async()
        self.responders: List[Responder] = [e for e, _ in agent_entity_responders]
        self.responders_async: List[Responder] = [
            e for e, _ in agent_entity_responders_async
        ]
        self.non_human_responders: List[Responder] = [
            r for r in self.responders if r != Entity.USER
        ]
        self.non_human_responders_async: List[Responder] = [
            r for r in self.responders_async if r != Entity.USER
        ]

        self.human_tried = False  # did human get a chance to respond in last step?
        self._entity_responder_map: Dict[
            Entity, Callable[..., Optional[ChatDocument]]
        ] = dict(agent_entity_responders)

        self._entity_responder_async_map: Dict[
            Entity, Callable[..., Coroutine[Any, Any, Optional[ChatDocument]]]
        ] = dict(agent_entity_responders_async)

        self.name_sub_task_map: Dict[str, Task] = {}
        # latest message in a conversation among entities and agents.
        self.pending_message: Optional[ChatDocument] = None
        self.pending_sender: Responder = Entity.USER
        self.single_round = single_round
        self.turns = -1  # no limit
        self.llm_delegate = llm_delegate
        if llm_delegate:
            if self.single_round:
                # 0: User instructs (delegating to LLM);
                # 1: LLM (as the Controller) asks;
                # 2: user replies.
                self.turns = 2
        else:
            if self.single_round:
                # 0: User (as Controller) asks,
                # 1: LLM replies.
                self.turns = 1
        # other sub_tasks this task can delegate to
        self.sub_tasks: List[Task] = []
        self.caller: Task | None = None  # which task called this task's `run` method

    def clone(self, i: int) -> "Task":
        """
        Returns a copy of this task, with a new agent.
        """
        assert isinstance(self.agent, ChatAgent), "Task clone only works for ChatAgent"
        agent: ChatAgent = self.agent.clone(i)
        return Task(
            agent,
            name=self.name + f"-{i}",
            llm_delegate=self.llm_delegate,
            single_round=self.single_round,
            system_message=self.agent.system_message,
            user_message=self.agent.user_message,
            restart=self.restart,
            default_human_response=self.default_human_response,
            interactive=self.interactive,
            erase_substeps=self.erase_substeps,
            allow_null_result=self.allow_null_result,
            max_stalled_steps=self.max_stalled_steps,
            done_if_no_response=[Entity(s) for s in self.done_if_no_response],
            done_if_response=[Entity(s) for s in self.done_if_response],
            config=self.config,
        )

    @classmethod
    def cache(cls) -> RedisCache:
        if cls._cache is None:
            cls._cache = RedisCache(RedisCacheConfig(fake=False))
        return cls._cache

    @classmethod
    def _start_background_tasks(cls) -> None:
        """Start background object registry cleanup thread. NOT USED."""
        if cls._background_tasks_started:
            return
        cls._background_tasks_started = True
        cleanup_thread = threading.Thread(
            target=scheduled_cleanup,
            args=(600,),
            daemon=True,
        )
        cleanup_thread.start()

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

    def _init_message_counter(self) -> None:
        self.message_counter.clear()
        # create a unique string that will not likely be in any message,
        # so we always have a message with count=1
        self.message_counter.update([hash("___NO_MESSAGE___")])

    def _cache_session_store(self, key: str, value: str) -> None:
        """
        Cache a key-value pair for the current session.
        E.g. key = "kill", value = "1"
        """
        try:
            self.cache().store(f"{self.session_id}:{key}", value)
        except Exception as e:
            logging.error(f"Error in Task._cache_session_store: {e}")

    def _cache_session_lookup(self, key: str) -> Dict[str, Any] | str | None:
        """
        Retrieve a value from the cache for the current session.
        """
        session_id_key = f"{self.session_id}:{key}"
        try:
            cached_val = self.cache().retrieve(session_id_key)
        except Exception as e:
            logging.error(f"Error in Task._cache_session_lookup: {e}")
            return None
        return cached_val

    def _is_kill(self) -> bool:
        """
        Check if the current session is killed.
        """
        return self._cache_session_lookup("kill") == "1"

    def _set_alive(self) -> None:
        """
        Initialize the kill status of the current session.
        """
        self._cache_session_store("kill", "0")

    @classmethod
    def kill_session(cls, session_id: str = "") -> None:
        """
        Kill the session with the given session_id.
        """
        session_id_kill_key = f"{session_id}:kill"
        cls.cache().store(session_id_kill_key, "1")

    def kill(self) -> None:
        """
        Kill the task run associated with the current session.
        """
        self._cache_session_store("kill", "1")

    @property
    def _level(self) -> int:
        if self.caller is None:
            return 0
        return self.caller._level + 1

    @property
    def _indent(self) -> str:
        return "...|" * self._level

    @property
    def _enter(self) -> str:
        return self._indent + ">>>"

    @property
    def _leave(self) -> str:
        return self._indent + "<<<"

    def add_sub_task(
        self,
        task: (
            Task | List[Task] | Tuple[Task, TaskConfig] | List[Tuple[Task, TaskConfig]]
        ),
    ) -> None:
        """
        Add a sub-task (or list of subtasks) that this task can delegate
        (or fail-over) to. Note that the sequence of sub-tasks is important,
        since these are tried in order, as the parent task searches for a valid
        response (unless a sub-task is explicitly addressed).

        Args:
            task: A task, or list of tasks, or a tuple of task and task config,
                or a list of tuples of task and task config.
                These tasks are added as sub-tasks of the current task.
                The task configs (if any) dictate how the tasks are run when
                invoked as sub-tasks of other tasks. This allows users to specify
                behavior applicable only in the context of a particular task-subtask
                combination.
        """
        if isinstance(task, list):
            for t in task:
                self.add_sub_task(t)
            return

        if isinstance(task, tuple):
            task, config = task
        else:
            config = TaskConfig()
        task.config_sub_task = config
        self.sub_tasks.append(task)
        self.name_sub_task_map[task.name] = task
        self.responders.append(cast(Responder, task))
        self.responders_async.append(cast(Responder, task))
        self.non_human_responders.append(cast(Responder, task))
        self.non_human_responders_async.append(cast(Responder, task))

    def init(self, msg: None | str | ChatDocument = None) -> ChatDocument | None:
        """
        Initialize the task, with an optional message to start the conversation.
        Initializes `self.pending_message` and `self.pending_sender`.
        Args:
            msg (str|ChatDocument): optional message to start the conversation.

        Returns:
            (ChatDocument|None): the initialized `self.pending_message`.
            Currently not used in the code, but provided for convenience.
        """
        self.pending_sender = Entity.USER
        if isinstance(msg, str):
            self.pending_message = ChatDocument(
                content=msg,
                metadata=ChatDocMetaData(
                    sender=Entity.USER,
                ),
            )
        elif msg is None and len(self.agent.message_history) > 1:
            # if agent has a history beyond system msg, set the
            # pending message to the ChatDocument linked from
            # last message in the history
            last_agent_msg = self.agent.message_history[-1]
            self.pending_message = ChatDocument.from_id(last_agent_msg.chat_document_id)
            if self.pending_message is not None:
                self.pending_sender = self.pending_message.metadata.sender
        else:
            if isinstance(msg, ChatDocument):
                # carefully deep-copy: fresh metadata.id, register
                # as new obj in registry
                self.pending_message = ChatDocument.deepcopy(msg)
            if self.pending_message is not None and self.caller is not None:
                # msg may have come from `caller`, so we pretend this is from
                # the CURRENT task's USER entity
                self.pending_message.metadata.sender = Entity.USER
                # update parent, child, agent pointers
                if msg is not None:
                    msg.metadata.child_id = self.pending_message.metadata.id
                    self.pending_message.metadata.parent_id = msg.metadata.id
                self.pending_message.metadata.agent_id = self.agent.id

        self._show_pending_message_if_debug()

        if self.caller is not None and self.caller.logger is not None:
            self.logger = self.caller.logger
        else:
            self.logger = RichFileLogger(
                str(Path(self.config.logs_dir) / f"{self.name}.log"),
                color=self.color_log,
            )

        if self.caller is not None and self.caller.tsv_logger is not None:
            self.tsv_logger = self.caller.tsv_logger
        else:
            self.tsv_logger = setup_file_logger(
                "tsv_logger",
                str(Path(self.config.logs_dir) / f"{self.name}.tsv"),
            )
            header = ChatDocLoggerFields().tsv_header()
            self.tsv_logger.info(f" \tTask\tResponder\t{header}")

        self.log_message(Entity.USER, self.pending_message)
        return self.pending_message

    def reset_all_sub_tasks(self) -> None:
        """
        Recursively reset message history & state of own agent and
        those of all sub-tasks.
        """
        self.agent.init_state()
        for t in self.sub_tasks:
            t.reset_all_sub_tasks()

    def __getitem__(self, return_type: type) -> Task:
        """Returns a (shallow) copy of `self` with a default return type."""
        clone = copy.copy(self)
        clone.default_return_type = return_type
        return clone

    @overload
    def run(  # noqa
        self,
        msg: Any = None,
        *,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
    ) -> Optional[ChatDocument]: ...  # noqa

    @overload
    def run(  # noqa
        self,
        msg: Any = None,
        *,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
        return_type: Type[T],
    ) -> Optional[T]: ...  # noqa

    def run(
        self,
        msg: Any = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
        return_type: Optional[Type[T]] = None,
    ) -> Optional[ChatDocument | T]:
        """Synchronous version of `run_async()`.
        See `run_async()` for details."""
        if allow_restart and (
            (self.restart and caller is None)
            or (self.config_sub_task.restart_as_subtask and caller is not None)
        ):
            # We are either at top level, with restart = True, OR
            # we are a sub-task with restart_as_subtask = True,
            # so reset own agent and recursively for all sub-tasks
            self.reset_all_sub_tasks()

        self.n_stalled_steps = 0
        self._no_answer_step = -5  # last step where the best explicit response was N/A
        # how many N/A alternations have we had so far? (for Inf loop detection)
        self.n_no_answer_alternations = 0
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.session_id = session_id
        self._set_alive()
        self._init_message_counter()
        self.history.clear()

        msg_input = self.agent.to_ChatDocument(msg, author_entity=Entity.USER)

        if (
            isinstance(msg_input, ChatDocument)
            and msg_input.metadata.recipient != ""
            and msg_input.metadata.recipient != self.name
        ):
            # this task is not the intended recipient so return None
            return None

        self._pre_run_loop(
            msg=msg_input,
            caller=caller,
            is_async=False,
        )
        # self.turns overrides if it is > 0 and turns not set (i.e. = -1)
        turns = self.turns if turns < 0 else turns
        i = 0
        while True:
            self._step_idx = i  # used in step() below
            self.step()
            done, status = self.done()
            if done:
                if self._level == 0 and not settings.quiet:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            max_turns = (
                min(turns, settings.max_turns)
                if turns > 0 and settings.max_turns > 0
                else max(turns, settings.max_turns)
            )
            if max_turns > 0 and i >= max_turns:
                # Important to distinguish between:
                # (a) intentional run for a
                #     fixed number of turns, where we expect the pending message
                #     at that stage to be the desired result, and
                # (b) hitting max_turns limit, which is not intentional, and is an
                #     exception, resulting in a None task result
                status = (
                    StatusCode.MAX_TURNS
                    if i == settings.max_turns
                    else StatusCode.FIXED_TURNS
                )
                break
            if (
                self.config.inf_loop_cycle_len > 0
                and i % self.config.inf_loop_cycle_len == 0
                and self._maybe_infinite_loop()
                or self.n_no_answer_alternations > self.config.inf_loop_wait_factor
            ):
                raise InfiniteLoopException(
                    """Possible infinite loop detected!
                    You can adjust infinite loop detection (or turn it off)
                    by changing the params in the TaskConfig passed to the Task 
                    constructor; see here:
                    https://langroid.github.io/langroid/reference/agent/task/#langroid.agent.task.TaskConfig
                    """
                )

        final_result = self.result(status)
        self._post_run_loop()
        if final_result is None:
            return None

        if return_type is None:
            return_type = self.default_return_type

        if return_type is not None and return_type != ChatDocument:
            return self.agent.from_ChatDocument(final_result, return_type)
        return final_result

    @overload
    async def run_async(  # noqa
        self,
        msg: Any = None,
        *,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
    ) -> Optional[ChatDocument]: ...  # noqa

    @overload
    async def run_async(  # noqa
        self,
        msg: Any = None,
        *,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
        return_type: Type[T],
    ) -> Optional[T]: ...  # noqa

    async def run_async(
        self,
        msg: Any = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
        allow_restart: bool = True,
        return_type: Optional[Type[T]] = None,
    ) -> Optional[ChatDocument | T]:
        """
        Loop over `step()` until task is considered done or `turns` is reached.
        Runs asynchronously.

        Args:
            msg (Any): initial *user-role* message to process; if None,
                the LLM will respond to its initial `self.task_messages`
                which set up and kick off the overall task.
                The agent tries to achieve this goal by looping
                over `self.step()` until the task is considered
                done; this can involve a series of messages produced by Agent,
                LLM or Human (User). Note that `msg`, if passed, is treated as
                message with role `user`; a "system" role message should not be
                passed here.
            turns (int): number of turns to run the task for;
                default is -1, which means run until task is done.
            caller (Task|None): the calling task, if any
            max_cost (float): max cost allowed for the task (default 0 -> no limit)
            max_tokens (int): max tokens allowed for the task (default 0 -> no limit)
            session_id (str): session id for the task
            allow_restart (bool): whether to allow restarting the task
            return_type (Optional[Type[T]]): desired final result type

        Returns:
            Optional[ChatDocument]: valid result of the task.
        """

        # Even if the initial "sender" is not literally the USER (since the task could
        # have come from another LLM), as far as this agent is concerned, the initial
        # message can be considered to be from the USER
        # (from the POV of this agent's LLM).

        if allow_restart and (
            (self.restart and caller is None)
            or (self.config_sub_task.restart_as_subtask and caller is not None)
        ):
            # We are either at top level, with restart = True, OR
            # we are a sub-task with restart_as_subtask = True,
            # so reset own agent and recursively for all sub-tasks
            self.reset_all_sub_tasks()

        self.n_stalled_steps = 0
        self._no_answer_step = -5  # last step where the best explicit response was N/A
        # how many N/A alternations have we had so far? (for Inf loop detection)
        self.n_no_answer_alternations = 0
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.session_id = session_id
        self._set_alive()
        self._init_message_counter()
        self.history.clear()

        msg_input = self.agent.to_ChatDocument(msg, author_entity=Entity.USER)

        if (
            isinstance(msg_input, ChatDocument)
            and msg_input.metadata.recipient != ""
            and msg_input.metadata.recipient != self.name
        ):
            # this task is not the intended recipient so return None
            return None

        self._pre_run_loop(
            msg=msg_input,
            caller=caller,
            is_async=False,
        )
        # self.turns overrides if it is > 0 and turns not set (i.e. = -1)
        turns = self.turns if turns < 0 else turns
        i = 0
        while True:
            self._step_idx = i  # used in step() below
            await self.step_async()
            await asyncio.sleep(0.01)  # temp yield to avoid blocking
            done, status = self.done()
            if done:
                if self._level == 0 and not settings.quiet:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            max_turns = (
                min(turns, settings.max_turns)
                if turns > 0 and settings.max_turns > 0
                else max(turns, settings.max_turns)
            )
            if max_turns > 0 and i >= max_turns:
                # Important to distinguish between:
                # (a) intentional run for a
                #     fixed number of turns, where we expect the pending message
                #     at that stage to be the desired result, and
                # (b) hitting max_turns limit, which is not intentional, and is an
                #     exception, resulting in a None task result
                status = (
                    StatusCode.MAX_TURNS
                    if i == settings.max_turns
                    else StatusCode.FIXED_TURNS
                )
                break
            if (
                self.config.inf_loop_cycle_len > 0
                and i % self.config.inf_loop_cycle_len == 0
                and self._maybe_infinite_loop()
                or self.n_no_answer_alternations > self.config.inf_loop_wait_factor
            ):
                raise InfiniteLoopException(
                    """Possible infinite loop detected!
                    You can adjust infinite loop detection (or turn it off)
                    by changing the params in the TaskConfig passed to the Task 
                    constructor; see here:
                    https://langroid.github.io/langroid/reference/agent/task/#langroid.agent.task.TaskConfig
                    """
                )

        final_result = self.result(status)
        self._post_run_loop()
        if final_result is None:
            return None

        if return_type is None:
            return_type = self.default_return_type

        if return_type is not None and return_type != ChatDocument:
            return self.agent.from_ChatDocument(final_result, return_type)
        return final_result

    def _pre_run_loop(
        self,
        msg: Optional[str | ChatDocument] = None,
        caller: None | Task = None,
        is_async: bool = False,
    ) -> None:
        self.caller = caller
        self.init(msg)
        # sets indentation to be printed prior to any output from agent
        self.agent.indent = self._indent
        self.message_history_idx = -1
        if isinstance(self.agent, ChatAgent):
            # mark where we are in the message history, so we can reset to this when
            # we are done with the task
            self.message_history_idx = (
                max(
                    len(self.agent.message_history),
                    len(self.agent.task_messages),
                )
                - 1
            )
        # TODO decide on whether or not to print, based on is_async
        llm_model = (
            "no-LLM" if self.agent.llm is None else self.agent.llm.config.chat_model
        )
        if not settings.quiet:
            print(
                f"[bold magenta]{self._enter} Starting Agent "
                f"{self.name} ({self.message_history_idx+1}) "
                f"{llm_model} [/bold magenta]"
            )

    def _post_run_loop(self) -> None:
        # delete all messages from our agent's history, AFTER the first incoming
        # message, and BEFORE final result message
        n_messages = 0
        if isinstance(self.agent, ChatAgent):
            if self.erase_substeps:
                # TODO I don't like directly accessing agent message_history. Revisit.
                # (Pchalasani)
                # Note: msg history will consist of:
                # - H: the original msg history, ending at idx= self.message_history_idx
                # - R: this agent's response, which presumably leads to:
                # - X: a series of back-and-forth msgs (including with agent's own
                #     responders and with sub-tasks)
                # - F: the final result message, from this agent.
                # Here we are deleting all of [X] from the agent's message history,
                # so that it simply looks as if the sub-tasks never happened.

                dropped = self.agent.message_history[
                    self.message_history_idx + 2 : n_messages - 1
                ]
                # first delete the linked ChatDocuments (and descendants) from
                # ObjectRegistry
                for msg in dropped:
                    ChatDocument.delete_id(msg.chat_document_id)
                # then delete the messages from the agent's message_history
                del self.agent.message_history[
                    self.message_history_idx + 2 : n_messages - 1
                ]
            n_messages = len(self.agent.message_history)
        if self.erase_substeps:
            for t in self.sub_tasks:
                # erase our conversation with agent of subtask t

                # erase message_history of agent of subtask t
                # TODO - here we assume that subtask-agents are
                # ONLY talking to the current agent.
                if isinstance(t.agent, ChatAgent):
                    t.agent.clear_history(0)
        if not settings.quiet:
            print(
                f"[bold magenta]{self._leave} Finished Agent "
                f"{self.name} ({n_messages}) [/bold magenta]"
            )

    def step(self, turns: int = -1) -> ChatDocument | None:
        """
        Synchronous version of `step_async()`. See `step_async()` for details.
        TODO: Except for the self.response() calls, this fn should be identical to
        `step_async()`. Consider refactoring to avoid duplication.
        """
        self.is_done = False
        parent = self.pending_message
        recipient = (
            ""
            if self.pending_message is None
            else self.pending_message.metadata.recipient
        )
        if not self._valid_recipient(recipient):
            logger.warning(f"Invalid recipient: {recipient}")
            error_doc = ChatDocument(
                content=f"Invalid recipient: {recipient}",
                metadata=ChatDocMetaData(
                    sender=Entity.AGENT,
                    sender_name=Entity.AGENT,
                ),
            )
            self._process_valid_responder_result(Entity.AGENT, parent, error_doc)
            return error_doc

        responders: List[Responder] = self.non_human_responders.copy()

        if (
            Entity.USER in self.responders
            and not self.human_tried
            and not self.agent.has_tool_message_attempt(self.pending_message)
        ):
            # Give human first chance if they haven't been tried in last step,
            # and the msg is not a tool-call attempt;
            # (When `interactive=False`, human is only allowed to respond only if
            #  if explicitly addressed)
            # This ensures human gets a chance to respond,
            #   other than to a LLM tool-call.
            # When there's a tool msg attempt we want the
            #  Agent to be the next responder; this only makes a difference in an
            #  interactive setting: LLM generates tool, then we don't want user to
            #  have to respond, and instead let the agent_response handle the tool.

            responders.insert(0, Entity.USER)

        found_response = False
        # (responder, result) from a responder who explicitly said NO_ANSWER
        no_answer_response: None | Tuple[Responder, ChatDocument] = None
        n_non_responders = 0
        for r in responders:
            self.is_pass_thru = False
            if not self._can_respond(r):
                n_non_responders += 1
                # create dummy msg for logging
                log_doc = ChatDocument(
                    content="[CANNOT RESPOND]",
                    metadata=ChatDocMetaData(
                        sender=r if isinstance(r, Entity) else Entity.USER,
                        sender_name=str(r),
                        recipient=recipient,
                    ),
                )
                # no need to register this dummy msg in ObjectRegistry
                ChatDocument.delete_id(log_doc.id())
                self.log_message(r, log_doc)
                if n_non_responders == len(responders):
                    # don't stay in this "non-response" loop forever
                    break
                continue
            self.human_tried = r == Entity.USER
            result = self.response(r, turns)
            if result and NO_ANSWER in result.content:
                no_answer_response = (r, result)
            self.is_done = self._is_done_response(result, r)
            self.is_pass_thru = PASS in result.content if result else False
            if self.valid(result, r):
                found_response = True
                assert result is not None
                self._process_valid_responder_result(r, parent, result)
                break
            else:
                self.log_message(r, result)
            if self.is_done:
                # skip trying other responders in this step
                break
        if not found_response:  # did not find a valid response
            if no_answer_response:
                # even though there was no valid response from anyone in this step,
                # if there was at least one who EXPLICITLY said NO_ANSWER, then
                # we process that as a valid response.
                r, result = no_answer_response
                self._process_valid_responder_result(r, parent, result)
            else:
                self._process_invalid_step_result(parent)
        self._show_pending_message_if_debug()
        return self.pending_message

    async def step_async(self, turns: int = -1) -> ChatDocument | None:
        """
        A single "turn" in the task conversation: The "allowed" responders in this
        turn (which can be either the 3 "entities", or one of the sub-tasks) are
        tried in sequence, until a _valid_ response is obtained; a _valid_
        response is one that contributes to the task, either by ending it,
        or producing a response to be further acted on.
        Update `self.pending_message` to the latest valid response (or NO_ANSWER
        if no valid response was obtained from any responder).

        Args:
            turns (int): number of turns to process. Typically used in testing
                where there is no human to "quit out" of current level, or in cases
                where we want to limit the number of turns of a delegated agent.

        Returns (ChatDocument|None):
            Updated `self.pending_message`. Currently the return value is not used
                by the `task.run()` method, but we return this as a convenience for
                other use-cases, e.g. where we want to run a task step by step in a
                different context.
        """
        self.is_done = False
        parent = self.pending_message
        recipient = (
            ""
            if self.pending_message is None
            else self.pending_message.metadata.recipient
        )
        if not self._valid_recipient(recipient):
            logger.warning(f"Invalid recipient: {recipient}")
            error_doc = ChatDocument(
                content=f"Invalid recipient: {recipient}",
                metadata=ChatDocMetaData(
                    sender=Entity.AGENT,
                    sender_name=Entity.AGENT,
                ),
            )
            self._process_valid_responder_result(Entity.AGENT, parent, error_doc)
            return error_doc

        responders: List[Responder] = self.non_human_responders_async.copy()

        if (
            Entity.USER in self.responders
            and not self.human_tried
            and not self.agent.has_tool_message_attempt(self.pending_message)
        ):
            # Give human first chance if they haven't been tried in last step,
            # and the msg is not a tool-call attempt;
            # This ensures human gets a chance to respond,
            #   other than to a LLM tool-call.
            # When there's a tool msg attempt we want the
            #  Agent to be the next responder; this only makes a difference in an
            #  interactive setting: LLM generates tool, then we don't want user to
            #  have to respond, and instead let the agent_response handle the tool.
            responders.insert(0, Entity.USER)

        found_response = False
        # (responder, result) from a responder who explicitly said NO_ANSWER
        no_answer_response: None | Tuple[Responder, ChatDocument] = None
        for r in responders:
            self.is_pass_thru = False
            if not self._can_respond(r):
                # create dummy msg for logging
                log_doc = ChatDocument(
                    content="[CANNOT RESPOND]",
                    metadata=ChatDocMetaData(
                        sender=r if isinstance(r, Entity) else Entity.USER,
                        sender_name=str(r),
                        recipient=recipient,
                    ),
                )
                # no need to register this dummy msg in ObjectRegistry
                ChatDocument.delete_id(log_doc.id())
                self.log_message(r, log_doc)
                continue
            self.human_tried = r == Entity.USER
            result = await self.response_async(r, turns)
            if result and NO_ANSWER in result.content:
                no_answer_response = (r, result)
            self.is_done = self._is_done_response(result, r)
            self.is_pass_thru = PASS in result.content if result else False
            if self.valid(result, r):
                found_response = True
                assert result is not None
                self._process_valid_responder_result(r, parent, result)
                break
            else:
                self.log_message(r, result)
            if self.is_done:
                # skip trying other responders in this step
                break
        if not found_response:
            if no_answer_response:
                # even though there was no valid response from anyone in this step,
                # if there was at least one who EXPLICITLY said NO_ANSWER, then
                # we process that as a valid response.
                r, result = no_answer_response
                self._process_valid_responder_result(r, parent, result)
            else:
                self._process_invalid_step_result(parent)
        self._show_pending_message_if_debug()
        return self.pending_message

    def _update_no_answer_vars(self, result: ChatDocument) -> None:
        """Update variables related to NO_ANSWER responses, to aid
        in alternating NO_ANSWER infinite-loop detection."""

        if NO_ANSWER in result.content:
            if self._no_answer_step == self._step_idx - 2:
                # N/A two steps ago
                self.n_no_answer_alternations += 1
            else:
                # reset alternations counter
                self.n_no_answer_alternations = 0

            # record the last step where the best explicit response was N/A
            self._no_answer_step = self._step_idx

    def _process_valid_responder_result(
        self,
        r: Responder,
        parent: ChatDocument | None,
        result: ChatDocument,
    ) -> None:
        """Processes valid result from a responder, during a step"""

        self._update_no_answer_vars(result)

        # pending_sender is of type Responder,
        # i.e. it is either one of the agent's entities
        # OR a sub-task, that has produced a valid response.
        # Contrast this with self.pending_message.metadata.sender, which is an ENTITY
        # of this agent, or a sub-task's agent.
        if not self.is_pass_thru:
            if self.pending_message is not None and not isinstance(r, Task):
                # when pending msg is from our own agent, respect the sender set there,
                # since sometimes a response may "mock" as if the response is from
                # another entity (e.g when using RewindTool, the agent handler
                # returns a result as if it were from the LLM).
                self.pending_sender = result.metadata.sender
            else:
                # when pending msg is from a sub-task, the sender is the sub-task
                self.pending_sender = r
            self.pending_message = result
        # set the parent/child links ONLY if not already set by agent internally,
        # which may happen when using the RewindTool, or in other scenarios.
        if parent is not None and not result.metadata.parent_id:
            result.metadata.parent_id = parent.id()
        if parent is not None and not parent.metadata.child_id:
            parent.metadata.child_id = result.id()

        self.log_message(self.pending_sender, result, mark=True)
        if self.is_pass_thru:
            self.n_stalled_steps += 1
        else:
            # reset stuck counter since we made progress
            self.n_stalled_steps = 0

        if self.pending_message is not None:
            if (
                self._is_done_response(result, r)
                and self._level == 0
                and self.only_user_quits_root
                and self._user_can_respond()
            ):
                # We're ignoring the DoneTools (if any) in this case,
                # so remove them from the pending msg, to ensure
                # they don't affect the next step.
                self.pending_message.tool_messages = [
                    t
                    for t in self.pending_message.tool_messages
                    if not isinstance(t, (DoneTool, AgentDoneTool))
                ]
            # update counters for infinite loop detection
            hashed_msg = hash(str(self.pending_message))
            self.message_counter.update([hashed_msg])
            self.history.append(hashed_msg)

    def _process_invalid_step_result(self, parent: ChatDocument | None) -> None:
        """
        Since step had no valid result from any responder, decide whether to update the
        self.pending_message to a NO_ANSWER message from the opposite entity,
        or leave it as is.
        Args:
           parent (ChatDocument|None): parent message of the current message
        """
        self.n_stalled_steps += 1
        if self.allow_null_result and not self.is_pass_thru:
            # Null step-result is allowed, and we're not in a "pass-thru" situation,
            # so we update the pending_message to a dummy NO_ANSWER msg
            # from the entity 'opposite' to the current pending_sender,
            # so that the task can continue.
            # CAUTION: unless the LLM is instructed to signal DONE at an appropriate
            # time, this can result in an infinite loop.
            responder = (
                Entity.LLM if self.pending_sender == Entity.USER else Entity.USER
            )
            parent_id = "" if parent is None else parent.id()
            self.pending_message = ChatDocument(
                content=NO_ANSWER,
                metadata=ChatDocMetaData(sender=responder, parent_id=parent_id),
            )
            self.pending_sender = responder
            self._update_no_answer_vars(self.pending_message)
        self.log_message(self.pending_sender, self.pending_message, mark=True)

    def _show_pending_message_if_debug(self) -> None:
        if self.pending_message is None:
            return
        if settings.debug:
            sender_str = escape(str(self.pending_sender))
            msg_str = escape(str(self.pending_message))
            print(f"[grey37][{sender_str}]{msg_str}[/grey37]")

    def _forbid_multi_oai_tools(self, e: Responder) -> ChatDocument:
        # Passing multiple OpenAI Tools to be handled by another agent
        # is not supported yet (we need to carefully establish correspondence
        # between the original tool-calls of agent A, and the returned results,
        # which may involve recursive-called tools by agent B).
        # So we set an error result corresponding to each tool-call.
        assert isinstance(
            e, Task
        ), "Forbidding multiple OAI tools only applies to a responder of type Task"
        err_str = """
                    ERROR: cannot pass multiple tools to another agent!
                    Please use ONE tool at a time!
                """
        id2result = OrderedDict((tc.id, err_str) for tc in self.agent.oai_tool_calls)
        result = e.agent.create_user_response(
            content="",
            oai_tool_id2result=id2result,
        )
        return result

    def response(
        self,
        e: Responder,
        turns: int = -1,
    ) -> Optional[ChatDocument]:
        """
        Sync version of `response_async()`. See `response_async()` for details.
        """
        if isinstance(e, Task):
            actual_turns = e.turns if e.turns > 0 else turns
            e.agent.callbacks.set_parent_agent(self.agent)
            # e.callbacks.set_parent_agent(self.agent)
            pending_tools = self.agent.try_get_tool_messages(self.pending_message)
            # TODO disable this
            if (
                len(pending_tools) > 1
                and len(self.agent.oai_tool_calls) > 1
                and not self.config.allow_subtask_multi_oai_tools
            ):
                result = self._forbid_multi_oai_tools(e)
            else:
                result = e.run(
                    self.pending_message,
                    turns=actual_turns,
                    caller=self,
                    max_cost=self.max_cost,
                    max_tokens=self.max_tokens,
                )
                # update result.tool_messages if any
                if isinstance(result, ChatDocument):
                    self.agent.try_get_tool_messages(result)
                if result is not None:
                    content, id2result, oai_tool_id = self.agent.process_tool_results(
                        result.content,
                        result.oai_tool_id2result,
                        (
                            self.pending_message.oai_tool_calls
                            if isinstance(self.pending_message, ChatDocument)
                            else None
                        ),
                    )
                    result.content = content
                    result.oai_tool_id2result = id2result
                    result.metadata.oai_tool_id = oai_tool_id

            result_str = (  # only used by callback to display content and possible tool
                "NONE"
                if result is None
                else "\n\n".join(str(m) for m in ChatDocument.to_LLMMessage(result))
            )
            maybe_tool = len(extract_top_level_json(result_str)) > 0
            self.callbacks.show_subtask_response(
                task=e,
                content=result_str,
                is_tool=maybe_tool,
            )
        else:
            response_fn = self._entity_responder_map[cast(Entity, e)]
            result = response_fn(self.pending_message)
            # update result.tool_messages if any
            if isinstance(result, ChatDocument):
                self.agent.try_get_tool_messages(result)

        result_chat_doc = self.agent.to_ChatDocument(
            result,
            chat_doc=self.pending_message,
            author_entity=e if isinstance(e, Entity) else Entity.USER,
        )
        return self._process_result_routing(result_chat_doc, e)

    def _process_result_routing(
        self, result: ChatDocument | None, e: Responder
    ) -> ChatDocument | None:
        # process result in case there is a routing instruction
        if result is None:
            return None
        if isinstance(result, ToolMessage):
            # this supports Agent responders and Task.run() to
            # return a ToolMessage, in addition str, ChatDocument
            if isinstance(e, Task):
                # With the curr defn of Task.result(),
                # Task.run() can't return a ToolMessage, so this case doesn't occur,
                # but we leave it here in case a
                # Task subclass overrides default behavior
                return e.agent.create_user_response(tool_messages=[result])
            else:
                # e must be this agent's Entity (LLM, AGENT or USER)
                return self.agent.response_template(e=e, tool_messages=[result])
        if not self.config.recognize_string_signals:
            # ignore all string-based signaling/routing
            return result
        # parse various routing/addressing strings in result
        is_pass, recipient, content = self._parse_routing(
            result,
            addressing_prefix=self.config.addressing_prefix,
        )
        if is_pass is None:  # no routing, i.e. neither PASS nor SEND
            return result
        if is_pass:
            if recipient is None or self.pending_message is None:
                # Just PASS, no recipient
                # This means pass on self.pending_message to the next responder
                # in the default sequence of responders.
                # So leave result intact since we handle "PASS" in step()
                return result
            # set recipient in self.pending_message
            self.pending_message.metadata.recipient = recipient
            # clear out recipient, replace with just PASS
            result.content = result.content.replace(
                f"{PASS_TO}:{recipient}", PASS
            ).strip()
            return result
        elif recipient is not None:
            # we are sending non-empty content to non-null recipient
            # clean up result.content, set metadata.recipient and return
            result.content = content or ""
            result.metadata.recipient = recipient
            return result
        else:
            return result

    async def response_async(
        self,
        e: Responder,
        turns: int = -1,
    ) -> Optional[ChatDocument]:
        """
        Get response to `self.pending_message` from a responder.
        If response is __valid__ (i.e. it ends the current turn of seeking
        responses):
            -then return the response as a ChatDocument object,
            -otherwise return None.
        Args:
            e (Responder): responder to get response from.
            turns (int): number of turns to run the task for.
                Default is -1, which means run until task is done.

        Returns:
            Optional[ChatDocument]: response to `self.pending_message` from entity if
            valid, None otherwise
        """
        if isinstance(e, Task):
            actual_turns = e.turns if e.turns > 0 else turns
            e.agent.callbacks.set_parent_agent(self.agent)
            pending_tools = self.agent.try_get_tool_messages(self.pending_message)
            # TODO disable this
            if (
                len(pending_tools) > 1
                and len(self.agent.oai_tool_calls) > 1
                and not self.config.allow_subtask_multi_oai_tools
            ):
                result = self._forbid_multi_oai_tools(e)
            else:
                # e.callbacks.set_parent_agent(self.agent)
                result = await e.run_async(
                    self.pending_message,
                    turns=actual_turns,
                    caller=self,
                    max_cost=self.max_cost,
                    max_tokens=self.max_tokens,
                )
                # update result.tool_messages if any
                if isinstance(result, ChatDocument):
                    self.agent.try_get_tool_messages(result)
                if result is not None:
                    content, id2result, oai_tool_id = self.agent.process_tool_results(
                        result.content,
                        result.oai_tool_id2result,
                        (
                            self.pending_message.oai_tool_calls
                            if isinstance(self.pending_message, ChatDocument)
                            else None
                        ),
                    )
                    result.content = content
                    result.oai_tool_id2result = id2result
                    result.metadata.oai_tool_id = oai_tool_id

            result_str = (  # only used by callback to display content and possible tool
                "NONE"
                if result is None
                else "\n\n".join(str(m) for m in ChatDocument.to_LLMMessage(result))
            )
            maybe_tool = len(extract_top_level_json(result_str)) > 0
            self.callbacks.show_subtask_response(
                task=e,
                content=result_str,
                is_tool=maybe_tool,
            )
        else:
            response_fn = self._entity_responder_async_map[cast(Entity, e)]
            result = await response_fn(self.pending_message)
            # update result.tool_messages if any
            if isinstance(result, ChatDocument):
                self.agent.try_get_tool_messages(result)

        result_chat_doc = self.agent.to_ChatDocument(
            result,
            chat_doc=self.pending_message,
            author_entity=e if isinstance(e, Entity) else Entity.USER,
        )
        return self._process_result_routing(result_chat_doc, e)

    def result(self, status: StatusCode | None = None) -> ChatDocument | None:
        """
        Get result of task. This is the default behavior.
        Derived classes can override this.

        Note the result of a task is returned as if it is from the User entity.

        Args:
            status (StatusCode): status of the task when it ended
        Returns:
            ChatDocument: result of task
        """
        if status in [StatusCode.STALLED, StatusCode.MAX_TURNS, StatusCode.INF_LOOP]:
            # In these case we don't know (and don't want to try to guess)
            # what the task result should be, so we return None
            return None

        result_msg = self.pending_message

        content = result_msg.content if result_msg else ""
        content_any = result_msg.content_any if result_msg else None
        if DONE in content and self.config.recognize_string_signals:
            # assuming it is of the form "DONE: <content>"
            content = content.replace(DONE, "").strip()
        oai_tool_calls = result_msg.oai_tool_calls if result_msg else None
        oai_tool_id2result = result_msg.oai_tool_id2result if result_msg else None
        fun_call = result_msg.function_call if result_msg else None
        tool_messages = result_msg.tool_messages if result_msg else []
        # if there is a DoneTool or AgentDoneTool among these,
        # we extract content and tools from here, and ignore all others
        for t in tool_messages:
            if isinstance(t, FinalResultTool):
                content = ""
                content_any = None
                tool_messages = [t]  # pass it on to parent so it also quits
                break
            elif isinstance(t, (AgentDoneTool, DoneTool)):
                # there shouldn't be multiple tools like this; just take the first
                content = to_string(t.content)
                content_any = t.content
                fun_call = None
                oai_tool_calls = None
                if isinstance(t, AgentDoneTool):
                    # AgentDoneTool may have tools, unlike DoneTool
                    tool_messages = t.tools
                break
        # drop the "Done" tools since they should not be part of the task result,
        # or else they would cause the parent task to get unintentionally done!
        tool_messages = [
            t for t in tool_messages if not isinstance(t, (DoneTool, AgentDoneTool))
        ]
        block = result_msg.metadata.block if result_msg else None
        recipient = result_msg.metadata.recipient if result_msg else ""
        tool_ids = result_msg.metadata.tool_ids if result_msg else []

        # regardless of which entity actually produced the result,
        # when we return the result, we set entity to USER
        # since to the "parent" task, this result is equivalent to a response from USER
        result_doc = ChatDocument(
            content=content,
            content_any=content_any,
            oai_tool_calls=oai_tool_calls,
            oai_tool_id2result=oai_tool_id2result,
            function_call=fun_call,
            tool_messages=tool_messages,
            metadata=ChatDocMetaData(
                source=Entity.USER,
                sender=Entity.USER,
                block=block,
                status=status or (result_msg.metadata.status if result_msg else None),
                sender_name=self.name,
                recipient=recipient,
                tool_ids=tool_ids,
                parent_id=result_msg.id() if result_msg else "",
                agent_id=str(self.agent.id),
            ),
        )
        if self.pending_message is not None:
            self.pending_message.metadata.child_id = result_doc.id()

        return result_doc

    def _is_empty_message(self, msg: str | ChatDocument | None) -> bool:
        """
        Check if msg is empty or None
        Args:
            msg (str|ChatDocument|None): message to check
        Returns:
            bool: True if msg is (equivalent to) empty or None, False otherwise
        """
        # if ignoring string-based signaling, set pass_str to ""
        pass_str = PASS if self.config.recognize_string_signals else ""
        return (
            msg is None
            or (isinstance(msg, str) and msg.strip() in [pass_str, ""])
            or (
                isinstance(msg, ChatDocument)
                and msg.content.strip() in [pass_str, ""]
                and msg.function_call is None
                and msg.oai_tool_calls is None
                and msg.oai_tool_id2result is None
                and msg.tool_messages == []
            )
        )

    def _is_done_response(
        self, result: str | None | ChatDocument, responder: Responder
    ) -> bool:
        """Is the task done based on the response from the given responder?"""

        allow_done_string = self.config.recognize_string_signals
        response_says_done = result is not None and (
            (isinstance(result, str) and DONE in result and allow_done_string)
            or (
                isinstance(result, ChatDocument)
                and (
                    (DONE in result.content and allow_done_string)
                    or (
                        any(
                            isinstance(t, (DoneTool, AgentDoneTool, FinalResultTool))
                            for t in result.tool_messages
                            # this condition ensures agent had chance to handle tools
                        )
                        and responder == Entity.AGENT
                    )
                )
            )
        )
        return (
            (
                responder.value in self.done_if_response
                and not self._is_empty_message(result)
            )
            or (
                responder.value in self.done_if_no_response
                and self._is_empty_message(result)
            )
            or (not self._is_empty_message(result) and response_says_done)
        )

    def _maybe_infinite_loop(self) -> bool:
        """
        Detect possible infinite loop based on message frequencies.
        NOTE: This detects two types of loops:
        - Alternating NO_ANSWER loops, specifically of the form
        x1 NO_ANSWER x2 NO_ANSWER x3 NO_ANSWER...
        (e.g. an LLM repeatedly saying something different, and another responder
        or sub-task saying NO_ANSWER -- i.e. "DO-NOT-KNOW")

        - "exact" loops, i.e. a cycle of messages that repeats exactly, e.g.
        a r b i t r a t e r a t e r a t e r a t e ...

        [It does not detect more general "approximate" loops, where two entities are
        responding to each other potentially forever, with (slightly) different
        messages each time]

        Here is the logic for the exact-loop detection:
        Intuition: when you look at a sufficiently long sequence with an m-message
        loop, then the frequencies of these m messages will "dominate" those
        of all other messages.

        1. First find m "dominant" messages, i.e. when arranged in decreasing
            frequency order, find the m such that
                freq[m] > F * freq[m+1] and
                freq[m] > W + freq[m+1]
            where F = config.inf_loop_dominance_factor (default 1.5) and
            W = config.inf_loop_wait_factor (default 5).
            So if you plot these frequencies in decreasing order,
            you will see a big drop in the plot, from m to m+1.
            We call the freqs until m the "dominant" freqs.
        2. Say we found m such dominant messages
           If the set of last (W * m) messages are the same as the
           set of m dominant messages,  then we are likely in a loop.
        """

        max_cycle_len = self.config.inf_loop_cycle_len
        if max_cycle_len <= 0:
            # no loop detection
            return False
        wait_factor = self.config.inf_loop_wait_factor
        if sum(self.message_counter.values()) < wait_factor * max_cycle_len:
            # we haven't seen enough messages to detect a loop
            return False

        # recall there's always a dummy msg with freq = 1
        most_common_msg_counts: List[Tuple[str, int]] = (
            self.message_counter.most_common(max_cycle_len + 1)
        )
        # get the most dominant msgs, i.e. these are at least 1.5x more freq
        # than the rest
        F = self.config.inf_loop_dominance_factor
        # counts array in non-increasing order
        counts = np.array([c for _, c in most_common_msg_counts])
        # find first index where counts[i] > F * counts[i+1]
        ratios = counts[:-1] / counts[1:]
        diffs = counts[:-1] - counts[1:]
        indices = np.where((ratios > F) & (diffs > wait_factor))[0]
        m = indices[-1] if indices.size > 0 else -1
        if m < 0:
            # no dominance found, but...
            if len(most_common_msg_counts) <= max_cycle_len:
                # ...The most-common messages are at most max_cycle_len,
                # even though we looked for the most common (max_cycle_len + 1) msgs.
                # This means there are only at most max_cycle_len distinct messages,
                # which also indicates a possible loop.
                m = len(most_common_msg_counts) - 1
            else:
                # ... we have enough messages, but no dominance found,
                # so there COULD be loops longer than max_cycle_len,
                # OR there is no loop at all; we can't tell, so we return False.
                return False

        dominant_msg_counts = most_common_msg_counts[: m + 1]
        # if the SET of dominant m messages is the same as the
        # the SET of last m*w messages, (where w = config.inf_loop_wait_factor),
        # then we are likely in a loop
        dominant_msgs = set([msg for msg, _ in dominant_msg_counts])
        lookback = wait_factor * (m + 1)
        recent_msgs = set(list(self.history)[-lookback:])
        return dominant_msgs == recent_msgs

    def done(
        self, result: ChatDocument | None = None, r: Responder | None = None
    ) -> Tuple[bool, StatusCode]:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Args:
            result (ChatDocument|None): result from a responder
            r (Responder|None): responder that produced the result
                Not used here, but could be used by derived classes.
        Returns:
            bool: True if task is done, False otherwise
            StatusCode: status code indicating why task is done
        """
        if self._is_kill():
            return (True, StatusCode.KILL)
        result = result or self.pending_message
        allow_done_string = self.config.recognize_string_signals
        # An entity decided task is done, either via DoneTool,
        # or by explicitly saying DONE
        done_result = result is not None and (
            (
                DONE in (result.content if isinstance(result, str) else result.content)
                and allow_done_string
            )
            or any(
                isinstance(t, (DoneTool, AgentDoneTool, FinalResultTool))
                for t in result.tool_messages
            )
        )

        user_quit = (
            result is not None
            and (result.content in USER_QUIT_STRINGS or done_result)
            and result.metadata.sender == Entity.USER
        )
        if self._level == 0 and self._user_can_respond() and self.only_user_quits_root:
            # for top-level task, only user can quit out
            return (user_quit, StatusCode.USER_QUIT if user_quit else StatusCode.OK)

        if self.is_done:
            return (True, StatusCode.DONE)

        if self.n_stalled_steps >= self.max_stalled_steps:
            # we are stuck, so bail to avoid infinite loop
            logger.warning(
                f"Task {self.name} stuck for {self.max_stalled_steps} steps; exiting."
            )
            return (True, StatusCode.STALLED)

        if self.max_cost > 0 and self.agent.llm is not None:
            try:
                if self.agent.llm.tot_tokens_cost()[1] > self.max_cost:
                    logger.warning(
                        f"Task {self.name} cost exceeded {self.max_cost}; exiting."
                    )
                    return (True, StatusCode.MAX_COST)
            except Exception:
                pass

        if self.max_tokens > 0 and self.agent.llm is not None:
            try:
                if self.agent.llm.tot_tokens_cost()[0] > self.max_tokens:
                    logger.warning(
                        f"Task {self.name} uses > {self.max_tokens} tokens; exiting."
                    )
                    return (True, StatusCode.MAX_TOKENS)
            except Exception:
                pass
        final = (
            # no valid response from any entity/agent in current turn
            result is None
            or done_result
            or (  # current task is addressing message to caller task
                self.caller is not None
                and self.caller.name != ""
                and result.metadata.recipient == self.caller.name
            )
            or user_quit
        )
        return (final, StatusCode.OK)

    def valid(
        self,
        result: Optional[ChatDocument],
        r: Responder,
    ) -> bool:
        """
        Is the result from a Responder (i.e. an entity or sub-task)
        such that we can stop searching for responses in this step?
        """
        # TODO caution we should ensure that no handler method (tool) returns simply
        # an empty string (e.g when showing contents of an empty file), since that
        # would be considered an invalid response, and other responders will wrongly
        # be given a chance to respond.

        # if task would be considered done given responder r's `result`,
        # then consider the result valid.
        if result is not None and self.done(result, r)[0]:
            return True
        return (
            result is not None
            and not self._is_empty_message(result)
            # some weaker LLMs, including even GPT-4o, may say "DO-NOT-KNOW."
            # (with a punctuation at the end), so need to strip out punctuation
            and re.sub(r"[,.!?:]", "", result.content.strip()) != NO_ANSWER
        )

    def log_message(
        self,
        resp: Responder,
        msg: ChatDocument | None = None,
        mark: bool = False,
    ) -> None:
        """
        Log current pending message, and related state, for lineage/debugging purposes.

        Args:
            resp (Responder): Responder that generated the `msg`
            msg (ChatDocument, optional): Message to log. Defaults to None.
            mark (bool, optional): Whether to mark the message as the final result of
                a `task.step()` call. Defaults to False.
        """
        default_values = ChatDocLoggerFields().dict().values()
        msg_str_tsv = "\t".join(str(v) for v in default_values)
        if msg is not None:
            msg_str_tsv = msg.tsv_str()

        mark_str = "*" if mark else " "
        task_name = self.name if self.name != "" else "root"
        resp_color = "white" if mark else "red"
        resp_str = f"[{resp_color}] {resp} [/{resp_color}]"

        if msg is None:
            msg_str = f"{mark_str}({task_name}) {resp_str}"
        else:
            color = {
                Entity.LLM: "green",
                Entity.USER: "blue",
                Entity.AGENT: "red",
                Entity.SYSTEM: "magenta",
            }[msg.metadata.sender]
            f = msg.log_fields()
            tool_type = f.tool_type.rjust(6)
            tool_name = f.tool.rjust(10)
            tool_str = f"{tool_type}({tool_name})" if tool_name != "" else ""
            sender = f"[{color}]" + str(f.sender_entity).rjust(10) + f"[/{color}]"
            sender_name = f.sender_name.rjust(10)
            recipient = "=>" + str(f.recipient).rjust(10)
            block = "X " + str(f.block or "").rjust(10)
            content = f"[{color}]{f.content}[/{color}]"
            msg_str = (
                f"{mark_str}({task_name}) "
                f"{resp_str} {sender}({sender_name}) "
                f"({recipient}) ({block}) {tool_str} {content}"
            )

        if self.logger is not None:
            self.logger.log(msg_str)
        if self.tsv_logger is not None:
            resp_str = str(resp)
            self.tsv_logger.info(f"{mark_str}\t{task_name}\t{resp_str}\t{msg_str_tsv}")

    def _valid_recipient(self, recipient: str) -> bool:
        """
        Is the recipient among the list of responders?
        Args:
            recipient (str): Name of recipient
        """
        if recipient == "":
            return True
        # native responders names are USER, LLM, AGENT,
        # and the names of subtasks are from Task.name attribute
        responder_names = [self.name.lower()] + [
            r.name.lower() for r in self.responders
        ]
        return recipient.lower() in responder_names

    def _recipient_mismatch(self, e: Responder) -> bool:
        """
        Is the recipient explicitly specified and does not match responder "e" ?
        """
        # Note that recipient could be specified as an Entity or a Task name
        return (
            self.pending_message is not None
            and (recipient := self.pending_message.metadata.recipient) != ""
            and not (recipient == e)  # case insensitive for entities
            and recipient != e.name
            and recipient != self.name  # case sensitive
        )

    def _user_can_respond(self) -> bool:
        return self.interactive or (
            # regardless of self.interactive, if a msg is explicitly addressed to
            # user, then wait for user response
            self.pending_message is not None
            and self.pending_message.metadata.recipient == Entity.USER
            and not self.agent.has_tool_message_attempt(self.pending_message)
        )

    def _can_respond(self, e: Responder) -> bool:
        user_can_respond = self._user_can_respond()

        if self.pending_sender == e or (e == Entity.USER and not user_can_respond):
            # sender is same as e (an entity cannot respond to its own msg),
            # or user cannot respond
            return False

        if self.pending_message is None:
            return True
        if isinstance(e, Task) and not e.agent.can_respond(self.pending_message):
            return False

        if self._recipient_mismatch(e):
            # Cannot respond if not addressed to this entity
            return False
        return self.pending_message.metadata.block != e

    def set_color_log(self, enable: bool = True) -> None:
        """
        Flag to enable/disable color logging using rich.console.
        In some contexts, such as Colab notebooks, we may want to disable color logging
        using rich.console, since those logs show up in the cell output rather than
        in the log file. Turning off this feature will still create logs, but without
        the color formatting from rich.console
        Args:
            enable (bool): value of `self.color_log` to set to,
                which will enable/diable rich logging

        """
        self.color_log = enable

    def _parse_routing(
        self,
        msg: ChatDocument | str,
        addressing_prefix: str = "",
    ) -> Tuple[bool | None, str | None, str | None]:
        """
        Parse routing instruction if any, of the form:
        PASS:<recipient>  (pass current pending msg to recipient)
        SEND:<recipient> <content> (send content to recipient)
        @<recipient> <content> (send content to recipient)
        Args:
            msg (ChatDocument|str|None): message to parse
            addressing_prefix (str): prefix to address other agents or entities,
                 (e.g. "@". See documentation of `TaskConfig` for details).
        Returns:
            Tuple[bool|None, str|None, str|None]:
                bool: true=PASS, false=SEND, or None if neither
                str: recipient, or None
                str: content to send, or None
        """
        # handle routing instruction-strings in result if any,
        # such as PASS, PASS_TO, or SEND

        msg_str = msg.content if isinstance(msg, ChatDocument) else msg
        if (
            self.agent.has_tool_message_attempt(msg)
            and not msg_str.startswith(PASS)
            and not msg_str.startswith(PASS_TO)
            and not msg_str.startswith(SEND_TO)
        ):
            # if there's an attempted tool-call, we ignore any routing strings,
            # unless they are at the start of the msg
            return None, None, None

        content = msg.content if isinstance(msg, ChatDocument) else msg
        content = content.strip()
        if PASS in content and PASS_TO not in content:
            return True, None, None
        if PASS_TO in content and content.split(":")[1] != "":
            return True, content.split(":")[1], None
        if (
            SEND_TO in content
            and (addressee_content := parse_addressed_message(content, SEND_TO))[0]
            is not None
        ):
            # Note this will discard any portion of content BEFORE SEND_TO.
            # TODO maybe make this configurable.
            (addressee, content_to_send) = addressee_content
            # if no content then treat same as PASS_TO
            if content_to_send == "":
                return True, addressee, None
            else:
                return False, addressee, content_to_send
        if (
            addressing_prefix != ""
            and addressing_prefix in content
            and (
                addressee_content := parse_addressed_message(content, addressing_prefix)
            )[0]
            is not None
        ):
            (addressee, content_to_send) = addressee_content
            # if no content then treat same as PASS_TO
            if content_to_send == "":
                return True, addressee, None
            else:
                return False, addressee, content_to_send

        return None, None, None
