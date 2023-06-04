import json
import logging
from abc import ABC
from contextlib import ExitStack
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type

from pydantic import BaseSettings, ValidationError
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from llmagent.agent.message import INSTRUCTION, AgentMessage
from llmagent.language_models.base import LanguageModel, LLMConfig
from llmagent.mytypes import DocMetaData, Document
from llmagent.parsing.json import extract_top_level_json
from llmagent.parsing.parser import Parser, ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.configuration import settings
from llmagent.vector_store.base import VectorStore, VectorStoreConfig

console = Console()

logger = logging.getLogger(__name__)


class Entity(str, Enum):
    AGENT = "Agent"
    LLM = "LLM"
    USER = "User"


class ChatDocMetaData(DocMetaData):
    sender: Entity
    usage: int = 0
    cached: bool = False
    displayed: bool = False


class ChatDocument(Document):
    metadata: ChatDocMetaData


USER_QUIT = ["q", "x", "quit", "exit", "bye"]
NO_ANSWER = "I don't know"
DONE = "DONE:"


class AgentConfig(BaseSettings):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """

    name: str = "LLM-Agent"
    debug: bool = False
    vecdb: Optional[VectorStoreConfig] = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    parsing: Optional[ParsingConfig] = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig()


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.dialog: List[Tuple[str, str]] = []  # seq of LLM (prompt, response) tuples
        self.handled_classes: Dict[str, Type[AgentMessage]] = {}
        self.default_human_response: Optional[str] = None

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb) if config.vecdb else None
        self.parser: Optional[Parser] = (
            Parser(config.parsing) if config.parsing else None
        )

        self.allowed_responders: Set[Entity] = set()
        self._entity_responder_map = {
            Entity.USER: self.user_response,
            Entity.LLM: self.llm_response,
            Entity.AGENT: self.agent_response,
        }
        # latest message in a conversation among entities and agents.
        self.pending_message: Optional[ChatDocument] = None
        self.single_round = False
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
        self.controller = Entity.USER  # default controller of a task
        # latest "valid" response from an entity regardless of content
        # self.current_response: Optional[ChatDocument] = None
        # other agents that can process messages
        self.other_agents: List[Agent] = []
        self.parent_agent: Optional[Agent] = None
        self.level: int = 0  # level of agent hiearchy, 0 is top level
        self.indent = "...|" * self.level
        self.enter = self.indent + ">>>"
        self.leave = self.indent + "<<<"

    def update_dialog(self, prompt: str, output: str) -> None:
        self.dialog.append((prompt, output))

    def get_dialog(self) -> List[Tuple[str, str]]:
        return self.dialog

    def enable_message(self, message_class: Type[AgentMessage]) -> None:
        """
        Enable an agent to act on a message of a specific type from LLM

        Args:
            message_class (Type[AgentMessage]): The message class to enable.
        """
        if not issubclass(message_class, AgentMessage):
            raise ValueError("message_class must be a subclass of AgentMessage")

        request = message_class.__fields__["request"].default
        self.handled_classes[request] = message_class

    def disable_message(self, message_class: Type[AgentMessage]) -> None:
        """
        Disable a message class from being handled by this Agent.

        Args:
            message_class (Type[AgentMessage]): The message class to disable.
        """
        if not issubclass(message_class, AgentMessage):
            raise ValueError("message_class must be a subclass of AgentMessage")

        request = message_class.__fields__["request"].default
        if request in self.handled_classes:
            del self.handled_classes[request]

    def json_format_rules(self) -> str:
        """
        Specification of JSON formatting rules, based on the currently enabled
        message classes.

        Returns:
            str: formatting rules
        """
        enabled_classes: List[Type[AgentMessage]] = list(self.handled_classes.values())
        if len(enabled_classes) == 0:
            return "You can ask questions in natural language."

        json_conditions = "\n\n".join(
            [
                msg_cls().request + ":\n" + msg_cls().purpose  # type: ignore
                for i, msg_cls in enumerate(enabled_classes)
            ]
        )
        return json_conditions

    def sample_multi_round_dialog(self) -> str:
        """
        Generate a sample multi-round dialog based on enabled message classes.
        Returns:
            str: The sample dialog string.
        """
        enabled_classes: List[Type[AgentMessage]] = list(self.handled_classes.values())
        # use at most 2 sample conversations, no need to be exhaustive;
        sample_convo = [
            msg_cls().usage_example()  # type: ignore
            for i, msg_cls in enumerate(enabled_classes)
            if i < 2
        ]
        return "\n\n".join(sample_convo)

    def message_format_instructions(self) -> str:
        """
        Generate a string containing instructions to the LLM on when to format
        requests/questions as JSON, based on the currently enabled message classes.

        Returns:
            str: The instructions string.
        """
        format_rules = self.json_format_rules()

        return f"""
        You have access to the following TOOLS to accomplish your task:
        TOOLS AVAILABLE:
        {format_rules}
        
        {INSTRUCTION}
        
        Now start, and be concise!                 
        """

    def agent_response(self, input_str: Optional[str] = None) -> Optional[ChatDocument]:
        if input_str is None:
            return None
        results = self.handle_message(input_str)
        if results is None:
            return None
        console.print(f"[red]{self.indent}", end="")
        print(f"[red]Agent: {results}")
        return ChatDocument(
            content=results,
            metadata=DocMetaData(
                source=Entity.AGENT.value,
                sender=Entity.AGENT.value,
            ),
        )

    def handle_message(self, input_str: str) -> Optional[str]:
        """
        Extract JSON substrings from input message, handle each by the appropriate
        handler method, and return the results as a combined string.

        Args:
            input_str (str): The input string possibly containing JSON.

        Returns:
            Optional[Str]: The result of the handler method in string form so it can
            be sent back to the LLM, or None if the input string was not successfully
            handled by a method.
        """
        json_substrings = extract_top_level_json(input_str)
        if len(json_substrings) == 0:
            return self.handle_message_fallback(input_str)
        results = [self._handle_one_json_message(j) for j in json_substrings]
        results_list = [r for r in results if r is not None]
        if len(results_list) == 0:
            return self.handle_message_fallback(input_str)
        return "\n".join(results_list)

    def handle_message_fallback(self, input_str: str) -> Optional[str]:
        """
        Fallback method to handle input string if no other handler method applies,
        or if an error is thrown.
        This method can be overridden by subclasses.

        Args:
            input_str (str): The input string.
        Returns:
            str: The result of the handler method in string form so it can
            be sent back to the LLM.
        """
        return None

    def _handle_one_json_message(self, json_str: str) -> Optional[str]:
        json_data = json.loads(json_str)
        request = json_data.get("request")
        if request is None:
            return None

        message_class = self.handled_classes.get(request)
        if message_class is None:
            logger.warning(f"No message class found for request '{request}'")
            return None

        try:
            message = message_class.parse_obj(json_data)
        except ValidationError as ve:
            raise ValueError("Error parsing JSON as message class") from ve

        handler_method = getattr(self, request, None)
        if handler_method is None:
            raise ValueError(f"No handler method found for request '{request}'")

        return handler_method(message)  # type: ignore

    def num_tokens(self, prompt: str) -> int:
        if self.parser is None:
            raise ValueError("Parser must be set, to count tokens")
        return self.parser.num_tokens(prompt)

    def user_response(self, msg: Optional[str] = None) -> Optional[ChatDocument]:
        """
        Get user response to current message.

        Args:
            msg (str): the current message


        Returns:
            User response, packaged as a ChatDocument

        """
        if self.default_human_response is not None:
            # useful for automated testing
            user_msg = self.default_human_response
        else:
            user_msg = Prompt.ask(
                f"[blue]{self.indent}Human "
                f"(respond or q, x to exit current level, "
                f"or hit enter to continue)\n{self.indent}",
            ).strip()

        # only return non-None result if user_msg not empty
        if not user_msg:
            return None
        else:
            return ChatDocument(
                content=user_msg,
                metadata=DocMetaData(
                    source=Entity.USER.value,
                    sender=Entity.USER.value,
                ),
            )

    def _disallow_responder(self, e: Entity) -> None:
        """
        Disallow a responder from responding to current message.
        Args:
            e (Entity): entity to disallow
        """
        self.allowed_responders.remove(e)

    def _allow_responder(self, e: Entity) -> None:
        """
        Allow a responder to respond to current message.
        Args:
            e (Entity): entity to allow
        """
        self.allowed_responders.add(e)

    def _is_allowed_responder(self, e: Entity) -> bool:
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
        self.allowed_responders = set(Entity)  # map(lambda e: e.value, Entity))

    def _allow_all_responders_except(self, e: Entity) -> None:
        """
        Allow all responders to respond to current message, except for `e`.
        Args:
            e (Entity): entity to disallow
        """
        self._allow_all_responders()
        self._disallow_responder(e)

    def _entity_response(self, e: Entity) -> Optional[ChatDocument]:
        """
        Get response to `self.pending_message` from an entity.
        If response is __valid__ (i.e. it ends the current round of seeking
        responses):
            -then return the response as a ChatDocument object,
            -otherwise return None.

        A __valid__ response is one that _ends_ the current round of
        processing the `self.pending_message`: Responses can be of 3 types:
        - no response or equivalent;
            (e.g. LLM says "I don't know", or human enters empty string)
            This is NOT considered a valid response, and we continue
            attempting to get a response from another entity, in this round.
        - response that says "quit" or "done" or equivalent;
            (e.g. LLM says "DONE:<result>" or human enters "q" or "x").
            This response IS valid, since this indicates we end the current
            round of `process_pending_message`, as well as future rounds in the current
            "task", and the conversation quits out of the current level,
            to the "calling agent" or back to the top level loop.
        - any other response with actual content; This is also considered a
            valid response, since it means we have an updated
            self.pending_message to process in the next round, and we end the
            search for a response in this round.

        Args:
            e (Entity): entity to get response from
        Returns:
            Optional[ChatDocument]: response to `self.pending_message` from entity if
            valid, None otherwise
        """
        result: Optional[ChatDocument] = None
        msg = None if self.pending_message is None else self.pending_message.content
        if self._is_allowed_responder(e):
            result = self._entity_responder_map[e](msg)
            self._disallow_responder(e)
            if (
                result is not None
                and result.content != ""
                and NO_ANSWER not in result.content
            ):
                # We have a fresh "meaningful" message, so
                # enable all but the current entity to respond, in next round.
                self._allow_all_responders_except(e)
            else:
                result = None
        return result

    def process_pending_message(self, rounds: int = -1) -> None:
        """
        Possibly update `self.pending_message`, which could be from ANY entity
        (e.g. LLM, human, another agent).
        Effectively, this constitutes 1 "turn" of a conversation;

        The semantics of this method are as follows:
            - The _possible responders_ to the `self.pending_message` consist of: all
                entities _except_ the sender of the `self.pending_message` in the
                current  agent, and all agents in `self.other_agents` (if any).
            - These responders are tried in order, until a _valid_ response is
                obtained. See `_entity_response` for definition of _valid_.
            - As soon as a _valid_ response is obtained, the search for a
                response is ended. Whether or not `self.pending_message` is updated
                depends on:
                - if the (valid) response is simply a "quit" message from the user,
                    then `self.pending_message` is _not_ updated to this response,
                    which means `self.pending_message` retains the _last meaningful
                    response_ from an entity or external agent.
                - in all other cases, `self.pending_message` is updated to this
                    response. This response is assumed to contain meaningful
                    from a native entity (LLM, user, agent) or external agent, and so
                    updating `self.pending_message` to this response maintains the
                    invariant that `self.pending_message` contains the _last meaningful
                    response_ from an entity or external agent.
            - If no valid response is obtained from any responder, then
                `self.pending_message` is kept as is, and `self.current_response` is
                set to None.

        Thus the invariant maintained by this method is that:
        - `self.pending_message` is always the latest _meaningful_ response over all
        calls to `process_pending_message` in the current `_task_loop` (or the starting
        value if no new meaningful response was found in any calls so far).

        Args:
            rounds (int): number of rounds to process. Typically used in testing
                where there is no human to "quit out" of current level, or in cases
                where we want to limit the number of rounds of a delegated agent.

        """
        pending_message = (
            None if self.pending_message is None else self.pending_message.content
        )
        pending_sender = (
            Entity.USER
            if self.pending_message is None
            else self.pending_message.metadata.sender
        )

        responder = Entity.USER if pending_sender == Entity.LLM else Entity.LLM

        result = (
            self._entity_response(Entity.AGENT)
            or self._entity_response(Entity.LLM)
            or self._entity_response(Entity.USER)
        )
        if result is None and pending_sender == self.controller:
            # No valid response from agent's "native" entities;
            # If sender of pending response is `self.controller`, then
            # sequentially try to get valid response from `self.other_agents`
            # with the original pending_message
            for a in self.other_agents:
                result = a.do_task(msg=pending_message, rounds=rounds)
                if (
                    result is not None
                    and NO_ANSWER not in result.content
                    and result.content != ""
                ):
                    break

        response = NO_ANSWER if result is None else result.content
        self.reset_pending_message(msg=response, ent=responder)
        if settings.debug:
            pending_message = (
                "" if self.pending_message is None else self.pending_message.content
            )
            print(f"[red]pending_message: {pending_message}")

    def _task_done(self) -> bool:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Returns:
            bool: True if task is done, False otherwise
        """
        return (
            # no valid response from any entity/agent in current round
            self.pending_message is None
            # LLM decided task is done
            or DONE in self.pending_message.content
            or (
                # Task controller is "stuck", has nothing to say
                NO_ANSWER in self.pending_message.content
                and self.pending_message.metadata.sender == self.controller
            )
            or (
                # user intervenes and says quit out of current task/level
                self.pending_message.content in USER_QUIT
                and self.pending_message.metadata.sender == Entity.USER
            )
        )

    def reset_pending_message(
        self, msg: Optional[str] = None, ent: Entity = Entity.USER
    ) -> None:
        """
        Set up pending message (the "task")  before entering processing loop.

        Args:
            msg: initial msg; optional, default is None
            ent: initial sender; optional, default is Entity.USER

        """
        self._allow_all_responders_except(ent)
        self.pending_message = (
            None
            if msg is None
            else ChatDocument(content=msg, metadata=DocMetaData(source=ent, sender=ent))
        )

    def task_result(self) -> ChatDocument:
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
        last_non_controller_message = (
            self.last_user_message
            if self.controller == Entity.LLM
            else self.last_llm_message
        )
        if self.single_round:
            content = last_non_controller_message.content
        else:
            content = last_controller_message.content
            if DONE in content:
                content = content.replace(DONE, "").strip()
            else:
                content = NO_ANSWER

        return ChatDocument(
            content=content,
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )

    def _task_loop(self, rounds: int = -1) -> Optional[ChatDocument]:
        """
        Loop over `process_pending_message` until `task_done()` is True, or until
        `rounds` is reached. In each call to `process_pending_message`, the
        `self.current_response` and `self.pending_message` are updated.
        See `process_pending_message` for details.

        Args:
            rounds: how many rounds to process; optional, default is None, in which
            case the loop continues until `task_done()` is True.

        Returns:
            Optional[ChatDocument]: the result of the task, as determined by
            `task_result`.

        """
        i = 0
        print(
            f"[bold magenta]{self.enter} Starting Agent "
            f"{self.config.name}[/bold magenta]"
        )
        while True:
            self.process_pending_message()
            if self.pending_message is not None:
                if self.pending_message.metadata.sender == Entity.LLM:
                    self.last_llm_message = self.pending_message
                else:
                    self.last_user_message = self.pending_message
            if self._task_done():
                if self.level == 0:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            if rounds > 0 and i >= rounds:
                break
        result = self.task_result()
        print(
            f"[bold magenta]{self.leave} Finished Agent "
            f"{self.config.name}[/bold magenta]"
        )
        return result

    def do_task(
        self,
        msg: Optional[str] = None,
        rounds: int = -1,
        llm_delegate: bool = False,
    ) -> Optional[ChatDocument]:
        """
        Loop over `process` until task is considered done.

        Args:
            msg (str): initial message to process; if None,
                the LLM will respond to the initial `self.task_messages`
                which set up the overall task.
                The agent tries to achieve this goal by looping
                over `self.process_current_message(msg)` until the task is considered
                done; this can involve a series of messages produced by Agent,
                LLM or Human (User).
            rounds (int): number of rounds to run the task for;
                default is None, which means run until task is done.
            llm_delegate (bool): whether to delegate control to LLM

        Returns:
            ChatDocument: valid response from the agent
        """

        # Even the initial "sender" is not literally the USER (since the task could
        # have come from another LLM), as far as this agent is concerned, the initial
        # message can be considered to be from the USER.
        if llm_delegate:
            self.controller = Entity.LLM
        self.reset_pending_message(msg)
        if self.parent_agent is not None:
            self.level = self.parent_agent.level + 1
        self.indent = "...|" * self.level
        self.enter = self.indent + ">>>"
        self.leave = self.indent + "<<<"
        # `self.rounds` may be set via `self.add_agent`
        if self.single_round:
            rounds = 2  # overrides rounds param above
        return self._task_loop(rounds)

    def add_agent(
        self,
        agent: "Agent",
        llm_delegate: bool = False,
        single_round: bool = False,
    ) -> None:
        """
        Add an agent to process pending message when others fail.
        Args:
            agent (Agent): agent to add
            llm_delegate (bool): whether to delegate control to LLM
            single_round (bool): whether to run for a single round
        """
        agent.parent_agent = self
        if llm_delegate:
            agent.controller = Entity.LLM
        if single_round:
            agent.single_round = True
        self.other_agents.append(agent)

    def llm_response(self, prompt: Optional[str] = None) -> Optional[ChatDocument]:
        """
        Respond to a prompt.
        Args:
            prompt (str): prompt string
        Returns:
            Response from LLM, packaged as a ChatDocument
        """
        if prompt is None:
            return None
        with ExitStack() as stack:  # for conditionally using rich spinner
            if not self.llm.get_stream():
                # show rich spinner only if not streaming!
                cm = console.status("LLM responding to message...")
                stack.enter_context(cm)
            output_len = self.config.llm.max_output_tokens
            if (
                self.num_tokens(prompt) + output_len
                > self.llm.completion_context_length()
            ):
                output_len = self.llm.completion_context_length() - self.num_tokens(
                    prompt
                )
                if output_len < self.config.llm.min_output_tokens:
                    raise ValueError(
                        """
                    Token-length of Prompt + Output is longer than the
                    completion context length of the LLM!
                    """
                    )
                else:
                    logger.warning(
                        f"""
                    Requested output length has been shorted to {output_len}
                    so that the total length of Prompt + Output is less than
                    the completion context length of the LLM. 
                    """
                    )
            if self.llm.get_stream():
                console.print(f"[green]{self.indent}", end="")
            response = self.llm.generate(prompt, output_len)
        displayed = False
        if not self.llm.get_stream() or response.cached:
            # we would have already displayed the msg "live" ONLY if
            # streaming was enabled, AND we did not find a cached response
            console.print(f"[green]{self.indent}", end="")
            print("[green]" + response.message)
            displayed = True

        return ChatDocument(
            content=response.message,
            metadata=DocMetaData(
                source=Entity.LLM.value,
                sender=Entity.LLM.value,
                usage=response.usage,
                displayed=displayed,
                cached=response.cached,
            ),
        )

    def ask_agent(
        self,
        agent: "Agent",
        request: str,
        no_answer: str = NO_ANSWER,
        user_confirm: bool = True,
    ) -> Optional[str]:
        """
        Send a request to another agent, possibly after confirming with the user.

        Args:
            agent (Agent): agent to ask
            request (str): request to send
            no_answer: expected response when agent does not know the answer
            gate_human: whether to gate the request with a human confirmation

        Returns:
            str: response from agent
        """
        agent_type = type(agent).__name__
        if user_confirm:
            user_response = Prompt.ask(
                f"""[magenta]Here is the request or message:
                {request}
                Should I forward this to {agent_type}?""",
                default="y",
                choices=["y", "n"],
            )
            if user_response not in ["y", "yes"]:
                return None
        answer = agent.llm_response(request)
        if answer != no_answer:
            return (f"{agent_type} says: " + str(answer)).strip()
        return None
