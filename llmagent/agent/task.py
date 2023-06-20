from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Type, cast

from rich import print

from llmagent.agent.base import Agent, ChatDocMetaData, ChatDocument, Entity
from llmagent.agent.chat_agent import ChatAgent
from llmagent.language_models.base import LLMMessage, Role
from llmagent.mytypes import DocMetaData
from llmagent.parsing.agent_chats import parse_message
from llmagent.utils.configuration import settings
from llmagent.utils.constants import DONE, NO_ANSWER, USER_QUIT

Responder = Entity | Type["Task"]


class Task:
    """
    Class to maintain state needed to run a __task__. A __task__ generally
    is associated with a goal, typically represented by the initial "priming"
    messages of the LLM. Various entities take turns responding to
    `pending_message`, which is updated with the latest response.
    Tasks can have sub-tasks. A task is finished when `done()` returns true, and the
    final result is `result()`, which is returned to the "calling task" (if any).
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
    ):
        """
        A task to be performed by an agent.
        Args:
            agent (Agent): agent to perform the task
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

        self.agent = agent
        self.name = name or agent.config.name
        self.default_human_response = default_human_response
        if default_human_response is not None:
            self.agent.default_human_response = default_human_response
        self.only_user_quits_root = only_user_quits_root
        self.allowed_responders: Set[Responder] = set()
        self.responders: List[Responder] = [
            Entity.AGENT,  # response to LLM wanting to use tool or function_call
            Entity.LLM,
            Entity.USER,
        ]
        self._entity_responder_map: Dict[
            Entity, Callable[..., Optional[ChatDocument]]
        ] = {
            Entity.AGENT: self.agent.agent_response,
            Entity.LLM: self.agent.llm_response,
            Entity.USER: self.agent.user_response,
        }
        self.name_sub_task_map: Dict[str, Task] = {}
        # latest message in a conversation among entities and agents.
        self.pending_message: Optional[ChatDocument] = None
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

        self.last_llm_message = ChatDocument(
            content="",
            metadata=ChatDocMetaData(
                sender=Entity.LLM,
            ),
        )
        self.last_user_message = ChatDocument(
            content="",
            metadata=ChatDocMetaData(
                sender=Entity.USER,
            ),
        )

        # other sub_tasks this task can delegate to
        self.sub_tasks: List[Task] = []
        self.parent_task: Optional[Task] = None

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

    def add_sub_task(self, task: Task) -> None:
        """
        Add a sub-task that this task can delegate to

        Args:
            task (Task): sub-task to add
        """
        task.parent_task = self
        self.sub_tasks.append(task)
        self.responders.append(cast(Responder, task))
        self.name_sub_task_map[task.name] = task

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

        self.reset_pending_message(msg)
        # sets indentation to be printed prior to any output from agent
        self.agent.indent = self._indent
        if self.default_human_response is not None:
            self.agent.default_human_response = self.default_human_response

        message_history_idx = -1
        if isinstance(self.agent, ChatAgent):
            # mark where we are in the message history, so we can reset to this when
            # we are done with the task
            message_history_idx = (
                max(len(self.agent.message_history), len(self.agent.task_messages)) - 1
            )

        i = 0
        print(
            f"[bold magenta]{self._enter} Starting Agent "
            f"{self.name} ({message_history_idx+1}) [/bold magenta]"
        )
        while True:
            self.step()
            if self.pending_message is not None:
                if self.pending_message.metadata.sender == Entity.LLM:
                    self.last_llm_message = self.pending_message
                else:
                    self.last_user_message = self.pending_message
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
            len(self.agent.message_history)
            del self.agent.message_history[message_history_idx + 2 : n_messages - 1]
            n_messages = len(self.agent.message_history)
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

    def step(self, turns: int = -1) -> None:
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

        """
        # sender of current pending message
        pending_sender = (
            Entity.USER
            if self.pending_message is None  # only at start, at root task
            else self.pending_message.metadata.sender
        )

        responder_name = ""
        for r in self.responders:
            result = self.response(r, turns)
            if self.valid(result):
                if isinstance(r, Task):
                    responder_name = r.name
                elif (
                    self.pending_message is not None
                    and self.pending_message.function_call is not None
                ):
                    # when the response was to a `function_call`, set the
                    # `sender_name` to the name of the function that was called,
                    # so that in the LLM message-list, the `name` field of the latest
                    # message is set to the name of the function that was called.
                    responder_name = self.pending_message.function_call.name
                break

        response: str | ChatDocument # to appease mypy
        if result is None:
            response = NO_ANSWER
            responder = Entity.USER if pending_sender == Entity.LLM else Entity.LLM
        else:
            if result.content != "":
                response = result.content
            elif result.function_call is not None:
                response = result
            else:
                response = NO_ANSWER
            responder = result.metadata.sender

        self.reset_pending_message(
            msg=response,
            ent=responder,
            sender_name=responder_name,
        )
        if settings.debug:
            pending_message = (
                "" if self.pending_message is None else self.pending_message.content
            )
            sender_name_str = f"({responder_name})" if responder_name != "" else ""
            print(f"[red][{responder} {sender_name_str}]: {pending_message}")

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
        if not self._is_allowed_responder(e):
            return None

        if isinstance(e, Task):
            actual_turns = e.turns if e.turns > 0 else turns
            return e.run(self.pending_message, turns=actual_turns)
        else:
            return self._entity_responder_map[cast(Entity, e)](self.pending_message)

    def reset_pending_message(
        self,
        msg: Optional[str | ChatDocument] = None,
        ent: Entity = Entity.USER,
        sender_name: str = "",
    ) -> None:
        """
        Set up pending message (the "task")  before continuing processing loop.

        Args:
            msg (str|FunctionCall): latest message sent by an entity; optional,
                default is None. Could either be a string, or a FunctionCall object
                from the LLM (e.g. OpenAI `function_call` API response).
            ent (Entity): sender of the pending msg; optional, default is Entity.USER
            sender_name (str): name of sender; optional, default is ""
                (Used to fill in "name" field in OpenAI message list)

        """
        self._allow_all_responders_except(ent)
        fun_call = None
        recipient_task_name = ""
        if isinstance(msg, ChatDocument):
            if msg.function_call is not None:
                fun_call = msg.function_call
                recipient_task_name = msg.function_call.to
            else:
                msg = msg.content

        if fun_call is None:
            recipient_task_name, text = (
                parse_message(msg) if msg is not None else ("", "")
            )

        if recipient_task_name:
            # When a recipient task is specified, we only allow that task to respond,
            # so disable any sub-task that is NOT the recipient task
            for task in self.sub_tasks:
                if task.name != recipient_task_name:
                    self._disallow_responder(
                        cast(Type["Task"], self.name_sub_task_map[task.name])
                    )
        if (
            msg is not None
            and ent == Entity.USER
            and len(self.agent.get_tool_messages(msg)) > 0
        ):
            # if there is a valid "tool" message (either JSON or via `function_call`)
            # from the USER (typically comes from an outside entity),
            # then LLM of THIS agent should NOT respond!
            self._disallow_responder(Entity.LLM)

        self.pending_message = (
            None
            if msg is None
            else ChatDocument(
                content=text,
                function_call=fun_call,
                metadata=ChatDocMetaData(
                    source=ent,
                    sender=ent,
                    sender_name=sender_name,
                    recipient=recipient_task_name,
                ),
            )
        )

    def result(self) -> ChatDocument:
        """
        Get result of task. This is the default behavior.
        Derived classes can override this.
        Returns:
            ChatDocument: result of task
        """
        last_controller_message = (
            self.last_llm_message
            if self.controller == Entity.LLM
            else self.last_user_message
        )
        if self.single_round:
            content = self.pending_message.content if self.pending_message else ""
        else:
            content = last_controller_message.content
            if DONE in content:
                content = content.replace(DONE, "").strip()

        # regardless of which entity actually produced the result,
        # when we return the result, we set entity to USER
        # since to the "parent" task, this result is equivalent to a response from USER
        return ChatDocument(
            content=content,
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
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
            and result.content != ""
            and (  # if NO_ANSWER is from controller, then it means
                # controller is stuck and we are done with task loop
                NO_ANSWER not in result.content
                or result.metadata.sender == self.controller
            )
        )

    def _disallow_responder(self, e: Responder) -> None:
        """
        Disallow a responder from responding to current message.
        Args:
            e (Entity): entity to disallow
        """
        self.allowed_responders.remove(e)

    def _allow_responder(self, e: Responder) -> None:
        """
        Allow a responder to respond to current message.
        Args:
            e (Entity): entity to allow
        """
        self.allowed_responders.add(e)

    def _is_allowed_responder(self, e: Responder) -> bool:
        """
        Check if a responder is allowed to respond to current message.
        Args:
            e (Entity): entity to check
        Returns:
            bool: True if allowed, False otherwise
        """
        return e in self.allowed_responders

    def _allow_all_responders(self) -> None:
        """
        Allow all responders to respond to current message.
        """
        self.allowed_responders = set(self.responders)

    def _allow_all_responders_except(self, e: Responder) -> None:
        """
        Allow all responders to respond to current message, except for `e`.
        Args:
            e (Entity): entity to disallow
        """
        self._allow_all_responders()
        self._disallow_responder(e)
