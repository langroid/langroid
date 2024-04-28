from __future__ import annotations

import copy
import logging
import re
from collections import Counter
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

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
from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.mytypes import Entity
from langroid.parsing.parse_json import extract_top_level_json
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

logger = logging.getLogger(__name__)

Responder = Entity | Type["Task"]


def noop_fn(*args: List[Any], **kwargs: Dict[str, Any]) -> None:
    pass


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
    cache: RedisCache = RedisCache(RedisCacheConfig(fake=False))

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
        only_user_quits_root: bool = False,
        erase_substeps: bool = False,
        allow_null_result: bool = True,
        max_stalled_steps: int = 5,
        done_if_no_response: List[Responder] = [],
        done_if_response: List[Responder] = [],
    ):
        """
        A task to be performed by an agent.

        Args:
            agent (Agent): agent associated with the task
            name (str): name of the task
            llm_delegate (bool):
                [Deprecated, not used; use `done_if_response`, `done_if_no_response`
                instead]
                Whether to delegate control to LLM; conceptually,
                the "controlling entity" is the one "seeking" responses to its queries,
                and has a goal it is aiming to achieve. The "controlling entity" is
                either the LLM or the USER. (Note within a Task there is just one
                LLM, and all other entities are proxies of the "User" entity).
            single_round (bool):
                [Deprecated: Use `done_if_response`, `done_if_no_response` instead].
                If true, task runs until one message by controller,
                and subsequent response by non-controller. If false, runs for the
                specified number of turns in `run`, or until `done()` is true.
                One run of step() is considered a "turn".
            system_message (str): if not empty, overrides agent's system_message
            user_message (str): if not empty, overrides agent's user_message
            restart (bool): if true, resets the agent's message history
            default_human_response (str): default response from user; useful for
                testing, to avoid interactive input from user.
                [Instead of this, setting `interactive` usually suffices]
            interactive (bool): if true, wait for human input after each non-human
                response (prevents infinite loop of non-human responses).
                Default is true. If false, then `default_human_response` is set to ""
            only_user_quits_root (bool): if true, only user can quit the root task.
                [This param is ignored & deprecated; Keeping for backward compatibility.
                Instead of this, setting `interactive` suffices]
            erase_substeps (bool): if true, when task completes, erase intermediate
                conversation with subtasks from this agent's `message_history`, and also
                erase all subtask agents' `message_history`.
                Note: erasing can reduce prompt sizes, but results in repetitive
                sub-task delegation.
            allow_null_result (bool): [Deprecated, may be removed in future.]
                If true, allow null (empty or NO_ANSWER)
                as the result of a step or overall task result.
                Optional, default is True.
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

        if isinstance(agent, ChatAgent) and len(agent.message_history) == 0 or restart:
            agent = cast(ChatAgent, agent)
            agent.clear_history(0)
            agent.clear_dialog()
            # possibly change the system and user messages
            if system_message:
                # we always have at least 1 task_message
                agent.set_system_message(system_message)
            if user_message:
                agent.set_user_message(user_message)
        self.max_cost: float = 0
        self.max_tokens: int = 0
        self.session_id: str = ""
        self.logger: None | RichFileLogger = None
        self.tsv_logger: None | logging.Logger = None
        self.color_log: bool = False if settings.notebook else True
        self.agent = agent
        self.step_progress = False  # progress in current step?
        self.n_stalled_steps = 0  # how many consecutive steps with no progress?
        self.max_stalled_steps = max_stalled_steps
        self.done_if_response = [r.value for r in done_if_response]
        self.done_if_no_response = [r.value for r in done_if_no_response]
        self.is_done = False  # is task done (based on response)?
        self.is_pass_thru = False  # is current response a pass-thru?
        self.task_progress = False  # progress in current task (since run or run_async)?
        if name:
            # task name overrides name in agent config
            agent.config.name = name
        self.name = name or agent.config.name
        self.value: str = self.name
        self.default_human_response = default_human_response
        if default_human_response is not None and default_human_response == "":
            interactive = False
        self.interactive = interactive
        self.message_history_idx = -1
        if interactive:
            only_user_quits_root = True
        else:
            default_human_response = default_human_response or ""
            only_user_quits_root = False
        if default_human_response is not None:
            self.agent.default_human_response = default_human_response
        if self.interactive:
            self.agent.default_human_response = None
        self.only_user_quits_root = only_user_quits_root
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
            self.controller = Entity.LLM
            if self.single_round:
                # 0: User instructs (delegating to LLM);
                # 1: LLM asks;
                # 2: user replies.
                self.turns = 2
        else:
            self.controller = Entity.USER
            if self.single_round:
                self.turns = 1  # 0: User asks, 1: LLM replies.

        # other sub_tasks this task can delegate to
        self.sub_tasks: List[Task] = []
        self.parent_task: Set[Task] = set()
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
            restart=False,
            default_human_response=self.default_human_response,
            interactive=self.interactive,
            erase_substeps=self.erase_substeps,
            allow_null_result=self.allow_null_result,
            max_stalled_steps=self.max_stalled_steps,
            done_if_no_response=[Entity(s) for s in self.done_if_no_response],
            done_if_response=[Entity(s) for s in self.done_if_response],
        )

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

    def _cache_session_store(self, key: str, value: str) -> None:
        """
        Cache a key-value pair for the current session.
        E.g. key = "kill", value = "1"
        """
        try:
            self.cache.store(f"{self.session_id}:{key}", value)
        except Exception as e:
            logging.error(f"Error in Task._cache_session_store: {e}")

    def _cache_session_lookup(self, key: str) -> Dict[str, Any] | str | None:
        """
        Retrieve a value from the cache for the current session.
        """
        session_id_key = f"{self.session_id}:{key}"
        try:
            cached_val = self.cache.retrieve(session_id_key)
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
        cls.cache.store(session_id_kill_key, "1")

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

    def add_sub_task(self, task: Task | List[Task]) -> None:
        """
        Add a sub-task (or list of subtasks) that this task can delegate
        (or fail-over) to. Note that the sequence of sub-tasks is important,
        since these are tried in order, as the parent task searches for a valid
        response.

        Args:
            task (Task|List[Task]): sub-task(s) to add
        """

        if isinstance(task, list):
            for t in task:
                self.add_sub_task(t)
            return
        assert isinstance(task, Task), f"added task must be a Task, not {type(task)}"

        task.parent_task.add(self)  # add myself to set of parent tasks of `task`
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
        else:
            self.pending_message = msg
            if self.pending_message is not None and self.caller is not None:
                # msg may have come from `caller`, so we pretend this is from
                # the CURRENT task's USER entity
                self.pending_message.metadata.sender = Entity.USER

        self._show_pending_message_if_debug()

        if self.caller is not None and self.caller.logger is not None:
            self.logger = self.caller.logger
        else:
            self.logger = RichFileLogger(f"logs/{self.name}.log", color=self.color_log)

        if self.caller is not None and self.caller.tsv_logger is not None:
            self.tsv_logger = self.caller.tsv_logger
        else:
            self.tsv_logger = setup_file_logger("tsv_logger", f"logs/{self.name}.tsv")
            header = ChatDocLoggerFields().tsv_header()
            self.tsv_logger.info(f" \tTask\tResponder\t{header}")

        self.log_message(Entity.USER, self.pending_message)
        return self.pending_message

    def run(
        self,
        msg: Optional[str | ChatDocument] = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
    ) -> Optional[ChatDocument]:
        """Synchronous version of `run_async()`.
        See `run_async()` for details."""
        self.task_progress = False
        self.n_stalled_steps = 0
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.session_id = session_id
        self._set_alive()

        assert (
            msg is None or isinstance(msg, str) or isinstance(msg, ChatDocument)
        ), f"msg arg in Task.run() must be None, str, or ChatDocument, not {type(msg)}"

        if (
            isinstance(msg, ChatDocument)
            and msg.metadata.recipient != ""
            and msg.metadata.recipient != self.name
        ):
            # this task is not the intended recipient so return None
            return None
        self._pre_run_loop(
            msg=msg,
            caller=caller,
            is_async=False,
        )
        # self.turns overrides if it is > 0 and turns not set (i.e. = -1)
        turns = self.turns if turns < 0 else turns
        i = 0
        while True:
            self.step()
            done, status = self.done()
            if done:
                if self._level == 0 and not settings.quiet:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            if turns > 0 and i >= turns:
                status = StatusCode.MAX_TURNS
                break

        final_result = self.result()
        if final_result is not None:
            final_result.metadata.status = status
        self._post_run_loop()
        return final_result

    async def run_async(
        self,
        msg: Optional[str | ChatDocument] = None,
        turns: int = -1,
        caller: None | Task = None,
        max_cost: float = 0,
        max_tokens: int = 0,
        session_id: str = "",
    ) -> Optional[ChatDocument]:
        """
        Loop over `step()` until task is considered done or `turns` is reached.
        Runs asynchronously.

        Args:
            msg (str|ChatDocument): initial message to process; if None,
                the LLM will respond to its initial `self.task_messages`
                which set up and kick off the overall task.
                The agent tries to achieve this goal by looping
                over `self.step()` until the task is considered
                done; this can involve a series of messages produced by Agent,
                LLM or Human (User).
            turns (int): number of turns to run the task for;
                default is -1, which means run until task is done.
            caller (Task|None): the calling task, if any
            max_cost (float): max cost allowed for the task (default 0 -> no limit)
            max_tokens (int): max tokens allowed for the task (default 0 -> no limit)
            session_id (str): session id for the task

        Returns:
            Optional[ChatDocument]: valid result of the task.
        """

        # Even if the initial "sender" is not literally the USER (since the task could
        # have come from another LLM), as far as this agent is concerned, the initial
        # message can be considered to be from the USER
        # (from the POV of this agent's LLM).
        self.task_progress = False
        self.n_stalled_steps = 0
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.session_id = session_id
        self._set_alive()

        if (
            isinstance(msg, ChatDocument)
            and msg.metadata.recipient != ""
            and msg.metadata.recipient != self.name
        ):
            # this task is not the intended recipient so return None
            return None
        self._pre_run_loop(
            msg=msg,
            caller=caller,
            is_async=True,
        )
        # self.turns overrides if it is > 0 and turns not set (i.e. = -1)
        turns = self.turns if turns < 0 else turns
        i = 0
        while True:
            await self.step_async()
            done, status = self.done()
            if done:
                if self._level == 0 and not settings.quiet:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            if turns > 0 and i >= turns:
                status = StatusCode.MAX_TURNS
                break

        final_result = self.result()
        if final_result is not None:
            final_result.metadata.status = status
        self._post_run_loop()
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
        if self.default_human_response is not None:
            self.agent.default_human_response = self.default_human_response

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
            "no-LLM"
            if self.agent.config.llm is None
            else self.agent.config.llm.chat_model
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
        self.step_progress = False
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
            # This ensures human gets a chance to respond,
            #   other than to a LLM tool-call.
            # When there's a tool msg attempt we want the
            #  Agent to be the next responder; this only makes a difference in an
            #  interactive setting: LLM generates tool, then we don't want user to
            #  have to respond, and instead let the agent_response handle the tool.

            responders.insert(0, Entity.USER)

        found_response = False
        for r in responders:
            self.is_pass_thru = False
            if not self._can_respond(r):
                # create dummy msg for logging
                log_doc = ChatDocument(
                    content="[CANNOT RESPOND]",
                    function_call=None,
                    metadata=ChatDocMetaData(
                        sender=r if isinstance(r, Entity) else Entity.USER,
                        sender_name=str(r),
                        recipient=recipient,
                    ),
                )
                self.log_message(r, log_doc)
                continue
            self.human_tried = r == Entity.USER
            result = self.response(r, turns)
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
        self.step_progress = False
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
        for r in responders:
            if not self._can_respond(r):
                # create dummy msg for logging
                log_doc = ChatDocument(
                    content="[CANNOT RESPOND]",
                    function_call=None,
                    metadata=ChatDocMetaData(
                        sender=r if isinstance(r, Entity) else Entity.USER,
                        sender_name=str(r),
                        recipient=recipient,
                    ),
                )
                self.log_message(r, log_doc)
                continue
            self.human_tried = r == Entity.USER
            result = await self.response_async(r, turns)
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
            self._process_invalid_step_result(parent)
        self._show_pending_message_if_debug()
        return self.pending_message

    def _process_valid_responder_result(
        self,
        r: Responder,
        parent: ChatDocument | None,
        result: ChatDocument,
    ) -> None:
        """Processes valid result from a responder, during a step"""

        # pending_sender is of type Responder,
        # i.e. it is either one of the agent's entities
        # OR a sub-task, that has produced a valid response.
        # Contrast this with self.pending_message.metadata.sender, which is an ENTITY
        # of this agent, or a sub-task's agent.
        if not self.is_pass_thru:
            self.pending_sender = r
        result.metadata.parent = parent
        if not self.is_pass_thru:
            self.pending_message = result
        self.log_message(self.pending_sender, result, mark=True)
        self.step_progress = True
        self.task_progress = True
        if self.is_pass_thru:
            self.n_stalled_steps += 1
        else:
            # reset stuck counter since we made progress
            self.n_stalled_steps = 0

    def _process_invalid_step_result(self, parent: ChatDocument | None) -> None:
        """
        Since step had no valid result from any responder, decide whether to update the
        self.pending_message to a NO_ANSWER message from the opposite entity,
        or leave it as is.
        Args:
            parent (ChatDocument|None): parent message of the current message
        """
        self.n_stalled_steps += 1
        if (not self.task_progress or self.allow_null_result) and not self.is_pass_thru:
            # There has been no progress at all in this task, so we
            # update the pending_message to a dummy NO_ANSWER msg
            # from the entity 'opposite' to the current pending_sender,
            # so we show "progress" and avoid getting stuck in an infinite loop.
            responder = (
                Entity.LLM if self.pending_sender == Entity.USER else Entity.USER
            )
            self.pending_message = ChatDocument(
                content=NO_ANSWER,
                metadata=ChatDocMetaData(sender=responder, parent=parent),
            )
            self.pending_sender = responder
        self.log_message(self.pending_sender, self.pending_message, mark=True)

    def _show_pending_message_if_debug(self) -> None:
        if self.pending_message is None:
            return
        if settings.debug:
            sender_str = escape(str(self.pending_sender))
            msg_str = escape(str(self.pending_message))
            print(f"[grey37][{sender_str}]{msg_str}[/grey37]")

    def _parse_routing(self, msg: ChatDocument | str) -> Tuple[bool | None, str | None]:
        """
        Parse routing instruction if any, of the form:
        PASS:<recipient>  (pass current pending msg to recipient)
        SEND:<recipient> <content> (send content to recipient)
        Args:
            msg (ChatDocument|str|None): message to parse
        Returns:
            Tuple[bool,str|None]:
                bool: true=PASS, false=SEND, or None if neither
                str: recipient, or None
        """
        # handle routing instruction in result if any,
        # of the form PASS=<recipient>
        content = msg.content if isinstance(msg, ChatDocument) else msg
        content = content.strip()
        if PASS in content and PASS_TO not in content:
            return True, None
        if PASS_TO in content and content.split(":")[1] != "":
            return True, content.split(":")[1]
        if SEND_TO in content and (send_parts := re.split(r"[,: ]", content))[1] != "":
            # assume syntax is SEND_TO:<recipient> <content>
            # or SEND_TO:<recipient>,<content> or SEND_TO:<recipient>:<content>
            recipient = send_parts[1].strip()
            # get content to send, clean out routing instruction, and
            # start from 1 char after SEND_TO:<recipient>,
            # because we expect there is either a blank or some other separator
            # after the recipient
            content_to_send = content.replace(f"{SEND_TO}{recipient}", "").strip()[1:]
            # if no content then treat same as PASS_TO
            if content_to_send == "":
                return True, recipient
            else:
                return False, recipient
        return None, None

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
            result = e.run(
                self.pending_message,
                turns=actual_turns,
                caller=self,
                max_cost=self.max_cost,
                max_tokens=self.max_tokens,
            )
            result_str = str(ChatDocument.to_LLMMessage(result))
            maybe_tool = len(extract_top_level_json(result_str)) > 0
            self.callbacks.show_subtask_response(
                task=e,
                content=result_str,
                is_tool=maybe_tool,
            )
        else:
            response_fn = self._entity_responder_map[cast(Entity, e)]
            result = response_fn(self.pending_message)
        return self._process_result_routing(result)

    def _process_result_routing(
        self, result: ChatDocument | None
    ) -> ChatDocument | None:
        # process result in case there is a routing instruction
        if result is None:
            return None
        is_pass, recipient = self._parse_routing(result)
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
            result.content = result.content.replace(
                f"{SEND_TO}:{recipient}", ""
            ).strip()
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
            # e.callbacks.set_parent_agent(self.agent)
            result = await e.run_async(
                self.pending_message,
                turns=actual_turns,
                caller=self,
                max_cost=self.max_cost,
                max_tokens=self.max_tokens,
            )
            result_str = str(ChatDocument.to_LLMMessage(result))
            maybe_tool = len(extract_top_level_json(result_str)) > 0
            self.callbacks.show_subtask_response(
                task=e,
                content=result_str,
                is_tool=maybe_tool,
            )
        else:
            response_fn = self._entity_responder_async_map[cast(Entity, e)]
            result = await response_fn(self.pending_message)
        return self._process_result_routing(result)

    def result(self) -> ChatDocument:
        """
        Get result of task. This is the default behavior.
        Derived classes can override this.
        Returns:
            ChatDocument: result of task
        """
        result_msg = self.pending_message

        content = result_msg.content if result_msg else ""
        if DONE in content:
            # assuming it is of the form "DONE: <content>"
            content = content.replace(DONE, "").strip()
        fun_call = result_msg.function_call if result_msg else None
        tool_messages = result_msg.tool_messages if result_msg else []
        block = result_msg.metadata.block if result_msg else None
        recipient = result_msg.metadata.recipient if result_msg else None
        responder = result_msg.metadata.parent_responder if result_msg else None
        tool_ids = result_msg.metadata.tool_ids if result_msg else []
        status = result_msg.metadata.status if result_msg else None

        # regardless of which entity actually produced the result,
        # when we return the result, we set entity to USER
        # since to the "parent" task, this result is equivalent to a response from USER
        return ChatDocument(
            content=content,
            function_call=fun_call,
            tool_messages=tool_messages,
            metadata=ChatDocMetaData(
                source=Entity.USER,
                sender=Entity.USER,
                block=block,
                status=status,
                parent_responder=responder,
                sender_name=self.name,
                recipient=recipient,
                tool_ids=tool_ids,
            ),
        )

    def _is_empty_message(self, msg: str | ChatDocument | None) -> bool:
        """
        Check if msg is empty or None
        Args:
            msg (str|ChatDocument|None): message to check
        Returns:
            bool: True if msg is (equivalent to) empty or None, False otherwise
        """
        return (
            msg is None
            or (isinstance(msg, str) and msg.strip() in [PASS, ""])
            or (
                isinstance(msg, ChatDocument)
                and msg.content.strip() in [PASS, ""]
                and msg.function_call is None
                and msg.tool_messages == []
            )
        )

    def _is_done_response(
        self, result: str | None | ChatDocument, responder: Responder
    ) -> bool:
        """Is the task done based on the response from the given responder?"""

        response_says_done = result is not None and (
            (isinstance(result, str) and DONE in result)
            or (isinstance(result, ChatDocument) and DONE in result.content)
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

    def _maybe_infinite_loop(self, history: int = 10) -> bool:
        """
        TODO Not currently used, until we figure out best way.
        Check if {NO_ANSWER}, empty answer, or a specific non-LLM msg occurs too
        often in history of pending messages -- this can be an indicator of a possible
        multi-step infinite loop that we should exit.
        (A single-step infinite loop is where individual steps don't show progress
        and are easy to detect via n_stalled_steps, but a multi-step infinite loop
        could show "progress" at each step, but can still be an infinite loop, e.g.
        if the steps are just alternating between two messages).
        """
        p = self.pending_message
        n_no_answers = 0
        n_empty_answers = 0
        counter: Counter[str] = Counter()
        # count number of NO_ANSWER and empty answers in last up to 10 messages
        # in ancestors of self.pending_message
        for _ in range(history):
            if p is None:
                break
            n_no_answers += p.content.strip() == NO_ANSWER
            n_empty_answers += p.content.strip() == "" and p.function_call is None
            if p.metadata.sender != Entity.LLM and PASS not in p.content:
                counter.update([p.metadata.sender + ":" + p.content])
            p = p.metadata.parent

        # freq of most common message in history
        high_freq = (counter.most_common(1) or [("", 0)])[0][1]
        # We deem this a potential infinite loop if:
        # - a specific non-LLM msg occurs too often, or
        # - a NO_ANSWER or empty answer occurs too often
        return max(high_freq, n_no_answers) > self.max_stalled_steps

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
        user_quit = (
            result is not None
            and result.content in USER_QUIT_STRINGS
            and result.metadata.sender == Entity.USER
        )
        if self._level == 0 and self.only_user_quits_root:
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
            # An entity decided task is done
            or DONE in result.content
            or (  # current task is addressing message to caller task
                self.caller is not None
                and self.caller.name != ""
                and result.metadata.recipient == self.caller.name
            )
            # or (
            #     # Task controller is "stuck", has nothing to say
            #     NO_ANSWER in result.content
            #     and result.metadata.sender == self.controller
            # )
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
            and result.content.strip() != NO_ANSWER
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
            and recipient not in (e.name, self.name)
        )

    def _can_respond(self, e: Responder) -> bool:
        if self.pending_sender == e:
            # Responder cannot respond to its own message
            return False
        if self.pending_message is None:
            return True
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
