from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Type, cast

from rich import print

from langroid.agent.base import Agent
from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import (
    ChatDocLoggerFields,
    ChatDocMetaData,
    ChatDocument,
)
from langroid.language_models.base import LLMMessage, Role
from langroid.mytypes import Entity
from langroid.utils.configuration import settings
from langroid.utils.constants import DONE, NO_ANSWER, USER_QUIT
from langroid.utils.logging import RichFileLogger, setup_file_logger

Responder = Entity | Type["Task"]


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

    def __init__(
        self,
        agent: Agent,
        name: str = "",
        llm_delegate: bool = False,
        single_round: bool = False,
        system_message: str = "",
        user_message: str = "",
        restart: bool = False,
        default_human_response: Optional[str] = None,
        only_user_quits_root: bool = True,
        erase_substeps: bool = False,
    ):
        """
        A task to be performed by an agent.

        Args:
            agent (Agent): agent associated with the task
            name (str): name of the task
            llm_delegate (bool): whether to delegate control to LLM; conceptually,
                the "controlling entity" is the one "seeking" responses to its queries,
                and has a goal it is aiming to achieve. The "controlling entity" is
                either the LLM or the USER. (Note within a Task there is just one
                LLM, and all other entities are proxies of the "User" entity).
            single_round (bool): If true, task runs until one message by controller,
                and subsequent response by non-controller. If false, runs for the
                specified number of turns in `run`, or until `done()` is true.
                One run of step() is considered a "turn".
            system_message (str): if not empty, overrides agent's `task_messages[0]`
            user_message (str): if not empty, overrides agent's `task_messages[1]`
            restart (bool): if true, resets the agent's message history
            default_human_response (str): default response from user; useful for
                testing, to avoid interactive input from user.
            only_user_quits_root (bool): if true, only user can quit the root task.
            erase_substeps (bool): if true, when task completes, erase intermediate
                conversation with subtasks from this agent's `message_history`, and also
                erase all subtask agents' `message_history`.
                Note: erasing can reduce prompt sizes, but results in repetitive
                sub-task delegation.
        """
        if isinstance(agent, ChatAgent) and len(agent.message_history) == 0 or restart:
            agent = cast(ChatAgent, agent)
            agent.message_history = []
            # possibly change the task messages
            if system_message:
                # we always have at least 1 task_message
                agent.task_messages[0].content = system_message
            if user_message:
                agent.task_messages.append(
                    LLMMessage(
                        role=Role.USER,
                        content=user_message,
                    )
                )
        self.logger: None | RichFileLogger = None
        self.tsv_logger: None | logging.Logger = None
        self.color_log: bool = True
        self.agent = agent
        self.name = name or agent.config.name
        self.default_human_response = default_human_response
        if default_human_response is not None:
            self.agent.default_human_response = default_human_response
        self.only_user_quits_root = only_user_quits_root
        self.erase_substeps = erase_substeps

        agent_entity_responders = agent.entity_responders()
        self.responders: List[Responder] = [e for e, _ in agent_entity_responders]
        self.non_human_responders: List[Responder] = [
            r for r in self.responders if r != Entity.USER
        ]
        self.human_tried = False  # did human get a chance to respond in last step?
        self._entity_responder_map: Dict[
            Entity, Callable[..., Optional[ChatDocument]]
        ] = dict(agent_entity_responders)

        self.name_sub_task_map: Dict[str, Task] = {}
        # latest message in a conversation among entities and agents.
        self.pending_message: Optional[ChatDocument] = None
        self.pending_sender: Responder = Entity.USER
        self.single_round = single_round
        self.turns = -1  # no limit
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
        self.parent_task: Optional[Task] = None

    def __repr__(self) -> str:
        return f"{self.name}"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def _level(self) -> int:
        if self.parent_task is None:
            return 0
        else:
            return self.parent_task._level + 1

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

        task.parent_task = self
        self.sub_tasks.append(task)
        self.name_sub_task_map[task.name] = task
        self.responders.append(cast(Responder, task))
        self.non_human_responders.append(cast(Responder, task))

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
            if self.pending_message is not None and self.parent_task is not None:
                # msg may have come from parent_task, so we pretend this is from
                # the CURRENT task's USER entity
                self.pending_message.metadata.sender = Entity.USER

        if self.parent_task is not None and self.parent_task.logger is not None:
            self.logger = self.parent_task.logger
        else:
            self.logger = RichFileLogger(f"logs/{self.name}.log", color=self.color_log)

        if self.parent_task is not None and self.parent_task.tsv_logger is not None:
            self.tsv_logger = self.parent_task.tsv_logger
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
    ) -> Optional[ChatDocument]:
        """
        Loop over `step()` until task is considered done or `turns` is reached.

        Args:
            msg (str|ChatDocument): initial message to process; if None,
                the LLM will respond to the initial `self.task_messages`
                which set up the overall task.
                The agent tries to achieve this goal by looping
                over `self.step()` until the task is considered
                done; this can involve a series of messages produced by Agent,
                LLM or Human (User).
            turns (int): number of turns to run the task for;
                default is -1, which means run until task is done.

        Returns:
            Optional[ChatDocument]: valid response from the agent
        """

        # Even if the initial "sender" is not literally the USER (since the task could
        # have come from another LLM), as far as this agent is concerned, the initial
        # message can be considered to be from the USER
        # (from the POV of this agent's LLM).

        if (
            isinstance(msg, ChatDocument)
            and msg.metadata.recipient != ""
            and msg.metadata.recipient != self.name
        ):
            # this task is not the intended recipient so return None
            return None

        self.init(msg)
        # sets indentation to be printed prior to any output from agent
        self.agent.indent = self._indent
        if self.default_human_response is not None:
            self.agent.default_human_response = self.default_human_response

        message_history_idx = -1
        if isinstance(self.agent, ChatAgent):
            # mark where we are in the message history, so we can reset to this when
            # we are done with the task
            message_history_idx = (
                max(
                    len(self.agent.message_history),
                    len(self.agent.task_messages),
                )
                - 1
            )

        i = 0
        print(
            f"[bold magenta]{self._enter} Starting Agent "
            f"{self.name} ({message_history_idx+1}) [/bold magenta]"
        )
        while True:
            self.step()
            if self.done():
                if self._level == 0:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            if turns > 0 and i >= turns:
                break
        final_result = self.result()
        # delete all messages from our agent's history, AFTER the first incoming
        # message, and BEFORE final result message
        n_messages = 0
        if isinstance(self.agent, ChatAgent):
            if self.erase_substeps:
                del self.agent.message_history[message_history_idx + 2 : n_messages - 1]
            n_messages = len(self.agent.message_history)
        if self.erase_substeps:
            for t in self.sub_tasks:
                # erase our conversation with agent of subtask t

                # erase message_history of agent of subtask t
                # TODO - here we assume that subtask-agents are
                # ONLY talking to the current agent.
                if isinstance(t.agent, ChatAgent):
                    t.agent.clear_history(0)
        print(
            f"[bold magenta]{self._leave} Finished Agent "
            f"{self.name} ({n_messages}) [/bold magenta]"
        )
        return final_result

    def step(self, turns: int = -1) -> ChatDocument | None:
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
        result = None
        parent = self.pending_message
        recipient = (
            ""
            if self.pending_message is None
            else self.pending_message.metadata.recipient
        )
        responders: List[Responder] = self.non_human_responders.copy()
        if Entity.USER in self.responders and not self.human_tried:
            # give human first chance if they haven't been tried in last step:
            # ensures human gets chance at each turn.
            responders.insert(0, Entity.USER)

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
            result = self.response(r, turns)
            if self.valid(result):
                assert result is not None
                self.pending_sender = r
                if result.metadata.parent_responder is not None and not isinstance(
                    r, Entity
                ):
                    # When result is from a sub-task, and `result.metadata` contains
                    # a non-null `parent_responder`, pretend this result was
                    # from the parent_responder, by setting `self.pending_sender`.
                    self.pending_sender = result.metadata.parent_responder
                    # Since we've just used the "pretend responder",
                    # clear out the pretend responder in metadata
                    # (so that it doesn't get used again)
                    result.metadata.parent_responder = None
                result.metadata.parent = parent
                old_attachment = (
                    self.pending_message.attachment if self.pending_message else None
                )
                self.pending_message = result
                # if result has no attachment, preserve the old attachment
                if result.attachment is None:
                    self.pending_message.attachment = old_attachment
                self.log_message(self.pending_sender, result, mark=True)
                break
            else:
                self.log_message(r, result)

        if not self.valid(result):
            responder = (
                Entity.LLM if self.pending_sender == Entity.USER else Entity.USER
            )
            self.pending_message = ChatDocument(
                content=NO_ANSWER,
                metadata=ChatDocMetaData(sender=responder, parent=parent),
            )
            self.pending_sender = responder
            self.log_message(self.pending_sender, self.pending_message, mark=True)

        if settings.debug:
            sender_str = str(self.pending_sender)
            msg_str = str(self.pending_message)
            print(f"[red][{sender_str}]{msg_str}")
        return self.pending_message

    def response(self, e: Responder, turns: int = -1) -> Optional[ChatDocument]:
        """
        Get response to `self.pending_message` from an entity.
        If response is __valid__ (i.e. it ends the current turn of seeking
        responses):
            -then return the response as a ChatDocument object,
            -otherwise return None.
        Args:
            e (Entity): entity to get response from
        Returns:
            Optional[ChatDocument]: response to `self.pending_message` from entity if
            valid, None otherwise
        """
        if isinstance(e, Task):
            actual_turns = e.turns if e.turns > 0 else turns
            return e.run(self.pending_message, turns=actual_turns)
        else:
            return self._entity_responder_map[cast(Entity, e)](self.pending_message)

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
        attachment = result_msg.attachment if result_msg else None
        block = result_msg.metadata.block if result_msg else None
        recipient = result_msg.metadata.recipient if result_msg else None
        responder = result_msg.metadata.parent_responder if result_msg else None

        # regardless of which entity actually produced the result,
        # when we return the result, we set entity to USER
        # since to the "parent" task, this result is equivalent to a response from USER
        return ChatDocument(
            content=content,
            function_call=fun_call,
            attachment=attachment,
            metadata=ChatDocMetaData(
                source=Entity.USER,
                sender=Entity.USER,
                block=block,
                parent_responder=responder,
                sender_name=self.name,
                recipient=recipient,
            ),
        )

    def done(self) -> bool:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Returns:
            bool: True if task is done, False otherwise
        """
        user_quit = (
            self.pending_message is not None
            and self.pending_message.content in USER_QUIT
            and self.pending_message.metadata.sender == Entity.USER
        )
        if self._level == 0 and self.only_user_quits_root:
            # for top-level task, only user can quit out
            return user_quit

        return (
            # no valid response from any entity/agent in current turn
            self.pending_message is None
            # LLM decided task is done
            or DONE in self.pending_message.content
            or (  # current task is addressing message to parent task
                self.parent_task is not None
                and self.parent_task.name != ""
                and self.pending_message.metadata.recipient == self.parent_task.name
            )
            or (
                # Task controller is "stuck", has nothing to say
                NO_ANSWER in self.pending_message.content
                and self.pending_message.metadata.sender == self.controller
            )
            or user_quit
        )

    def valid(self, result: Optional[ChatDocument]) -> bool:
        """
        Is the result from an entity or sub-task such that we can stop searching
        for responses for this turn?
        """
        # TODO caution we should ensure that no handler method (tool) returns simply
        # an empty string (e.g when showing contents of an empty file), since that
        # would be considered an invalid response, and other responders will wrongly
        # be given a chance to respond.
        return (
            result is not None
            and (result.content != "" or result.function_call is not None)
            and (  # if NO_ANSWER is from controller, then it means
                # controller is stuck and we are done with task loop
                NO_ANSWER not in result.content
                or result.metadata.sender == self.controller
            )
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

    def _can_respond(self, e: Responder) -> bool:
        if self.pending_sender == e:
            return False
        if self.pending_message is None:
            return True
        if self.pending_message.metadata.block == e:
            # the entity should only be blocked at the first try;
            # Remove the block so it does not block the entity forever
            self.pending_message.metadata.block = None
            return False
        return self.pending_message is None or self.pending_message.metadata.block != e

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
