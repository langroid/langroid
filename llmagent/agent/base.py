from abc import ABC
from typing import Dict, Optional, Type, List, Tuple, Set
from contextlib import ExitStack
from pydantic import BaseSettings, ValidationError
from llmagent.mytypes import Document, DocMetaData
from rich import print
import json
from llmagent.agent.message import AgentMessage, INSTRUCTION
from llmagent.language_models.base import LanguageModel
from llmagent.vector_store.base import VectorStore
from llmagent.parsing.parser import Parser
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.parsing.json import extract_top_level_json
from llmagent.prompts.prompts_config import PromptsConfig
import logging
from rich.console import Console
from rich.prompt import Prompt
from enum import Enum

console = Console()

logger = logging.getLogger(__name__)


class Entity(str, Enum):
    AGENT = "Agent"
    LLM = "LLM"
    USER = "User"


USER_QUIT = ["q", "x", "quit", "exit", "bye"]
LLM_NO_ANSWER = "I don't know"
LLM_DONE = "DONE"


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
        self.dialog = []  # seq of LLM (prompt, response) tuples
        self.response: Document = None  # last response
        self.handled_classes: Dict[str, Type[AgentMessage]] = {}
        self.default_human_response = None

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb) if config.vecdb else None
        self.parser = Parser(config.parsing) if config.parsing else None

        self.allowed_responders: Set[Entity] = None
        self._entity_responder_map = {
            Entity.USER: self.user_response,
            Entity.LLM: self.llm_response,
            Entity.AGENT: self.agent_response,
        }
        # latest "meaningful" message that needs a response, i.e.
        # not "I don't know", or quit or done, etc.
        self.pending_message: Document = None
        # latest response from an entity regardless of content
        self.current_response: Document = None
        # other agents that can process messages
        self.other_agents: List[Agent] = []
        self.parent_agent: Agent = None
        self.level: int = 0  # level of agent hiearchy, 0 is top level

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
        enabled_classes: List[Type[AgentMessage]] = self.handled_classes.values()
        if len(enabled_classes) == 0:
            return "You can ask questions in natural language."

        json_conditions = "\n\n".join(
            [
                f""" 
                {msg_cls().request}: 
                {msg_cls().purpose}
                """
                # """
                # For example:
                # {msg_cls().usage_example()}
                # """
                for i, msg_cls in enumerate(enabled_classes)
            ]
        )
        return json_conditions

    def sample_multi_round_dialog(self):
        """
        Generate a sample multi-round dialog based on enabled message classes.
        Returns:
            str: The sample dialog string.
        """
        enabled_classes: List[Type[AgentMessage]] = self.handled_classes.values()
        # use at most 2 sample conversations, no need to be exhaustive;
        sample_convo = [
            msg_cls().usage_example()
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

    def agent_response(self, input_str: str = None) -> Document:
        if input_str is None:
            return None
        results = self.handle_message(input_str)
        if results is None:
            return None
        print(f"[red]Agent: {results}")
        return Document(
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
        results = [r for r in results if r is not None]
        if len(results) == 0:
            return self.handle_message_fallback(input_str)
        return "\n".join(results)

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

        return handler_method(message)

    def num_tokens(self, prompt: str) -> int:
        return self.parser.num_tokens(prompt)

    def user_response(self, msg: str = None) -> Optional[Document]:
        """
        Get user response to current message.

        Args:
            msg (str): the current message


        Returns:
            User response, packaged as a Document

        """
        if self.default_human_response is not None:
            # useful for automated testing
            user_msg = self.default_human_response
        else:
            user_msg = Prompt.ask(
                "[blue]Human "
                "(write response/request, q or x to exit current level, "
                "or hit enter to continue)\n",
            ).strip()

        # only return non-None result if user_msg not empty
        if user_msg:
            return Document(
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
        self.allowed_responders = set(map(lambda e: e.value, Entity))

    def _allow_all_responders_except(self, e: Entity) -> None:
        """
        Allow all responders to respond to current message, except for `e`.
        Args:
            e (Entity): entity to disallow
        """
        self._allow_all_responders()
        self._disallow_responder(e)

    def _entity_response(self, e: Entity) -> Optional[Document]:
        """
        Get response to current message, from an entity.
        Args:
            e (Entity): entity to get response from
        Returns:
            Optional[Document]: response from entity
        """
        msg = None if self.pending_message is None else self.pending_message.content
        if self._is_allowed_responder(e):
            result: Document = self._entity_responder_map[e](msg)
            self._disallow_responder(e)
            if result is not None and LLM_NO_ANSWER not in result.content:
                # We have a fresh "meaningful" message, so
                # enable all but the current entity to respond
                self._allow_all_responders_except(e)
                return result

    def process_pending_message(self, rounds: int = None) -> None:
        """
        Process pending message, which could be from ANY entity
        (e.g. LLM, human, another agent), to get a response.
        Effectively, this constitutes 1 "turn" of a conversation,
        e.g. if msg is from the LLM, the returned Document represents
        the user's response to the LLM's message, or if
        msg is from a human, the returned Document represents
        the LLM's response to the human's message.

        Args:
            rounds (int): number of rounds to process. Typically used in testing
                where there is no human to "quit out" of current level, or in cases
                where we want to limit the number of rounds of a delegated agent.

        Returns:
            Document: response to message
        """
        result = (
            self._entity_response(Entity.AGENT)
            or self._entity_response(Entity.LLM)
            or self._entity_response(Entity.USER)
        )
        native_result = result
        # sequentially try to get response from other agents,
        # each one picks up `pending_message` from the previous agent
        if result is None:
            pending_message = (
                None if self.pending_message is None else self.pending_message.content
            )
            pending_sender = (
                None
                if self.pending_message is None
                else self.pending_message.metadata.sender
            )
            for a in self.other_agents:
                result = a.do_task(msg=pending_message, rounds=rounds)
                if result is not None:
                    break
                pending_message = (
                    None if a.pending_message is None else a.pending_message.content
                )

        self.current_response = result
        if result is not None:
            if result.content not in USER_QUIT:
                self.pending_message = result
            if native_result is None:
                # we got `result` from another agent so
                # set up the state with the right entity:
                # if our pending msg (before delegating to other agents)
                # was sent by LLM, then we act as if this result is produced by USER,
                # and vice versa.
                ent = Entity.USER if pending_sender == Entity.LLM else Entity.LLM
                self.pending_message.metadata.sender = ent
                self.current_response.metadata.sender = ent
                self.setup_task(msg=result.content, ent=ent)

    def task_done(self) -> bool:
        """
        Check if task is done. This is the default behavior.
        Derived classes can override this.
        Returns:
            bool: True if task is done, False otherwise
        """
        return (
            self.current_response is None
            or self.current_response.content in USER_QUIT
            or (
                self.pending_message is not None
                and "DONE" in self.pending_message.content
            )
        )

    def setup_task(self, msg: str = None, ent: Entity = Entity.USER) -> None:
        """
        Set up task before entering processing loop.

        Args:
            msg: initial msg; optional, default is None
            ent: initial sender; optional, default is Entity.USER

        """
        self._allow_all_responders_except(ent)
        self.pending_message = (
            None
            if msg is None
            else Document(content=msg, metadata=DocMetaData(source=ent, sender=ent))
        )

    def _task_loop(self, rounds: int = None) -> Optional[Document]:
        i = 0
        indent = "...|" * self.level
        enter = indent + ">>>"
        leave = indent + "<<<"
        print(f"[bold magenta]{enter} Starting Agent {self.config.name}[/bold magenta]")

        while True:
            self.process_pending_message()
            if self.task_done():
                if self.level == 0:
                    print("[magenta]Bye, hope this was useful!")
                break
            i += 1
            if rounds is not None and i >= rounds:
                break
        print(f"[bold magenta]{leave} Finished Agent {self.config.name}[/bold magenta]")
        return self.task_result()

    def do_task(self, msg: str = None, rounds: int = None) -> Optional[Document]:
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

        Returns:
            Document: final response from the agent
        """

        # Even the initial "sender" is not literally the USER (since the task could
        # have come from another LLM), as far as this agent is concerned, the initial
        # message can be considered to be from the USER.
        self.setup_task(msg)
        if self.parent_agent is not None:
            self.level = self.parent_agent.level + 1
        return self._task_loop(rounds)

    def add_agent(self, agent: "Agent"):
        """
        Add an agent to process pending message when others fail.
        Args:
            agent (Agent): agent to add
        """
        agent.parent_agent = self
        self.other_agents.append(agent)

    def task_result(self) -> Optional[Document]:
        """
        Get result of task. This is the default behavior.
        Derived classes can override this.
        Returns:
            Document: result of task
        """
        return Document(
            content=self.pending_message.content.replace("DONE:", "").strip(),
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )

    def llm_response(self, prompt: str) -> Document:
        """
        Respond to a prompt.
        Args:
            prompt (str): prompt string
        Returns:
            Response from LLM, packaged as a Document
        """

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

            response = self.llm.generate(prompt, output_len)
        displayed = False
        if not self.llm.get_stream() or response.cached:
            # we would have already displayed the msg "live" ONLY if
            # streaming was enabled, AND we did not find a cached response
            print("[green]" + response.message)
            displayed = True

        return Document(
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
        no_answer: str = LLM_NO_ANSWER,
        user_confirm: bool = True,
    ) -> Optional[Document]:
        """
        Send a request to another agent, possibly after confirming with the user.

        Args:
            agent (Agent): agent to ask
            request (str): request to send
            no_answer: expected response when agent does not know the answer
            gate_human: whether to gate the request with a human confirmation

        Returns:
            Document: response from agent
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
