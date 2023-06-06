import json
import logging
from abc import ABC
from contextlib import ExitStack
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

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
from llmagent.utils.constants import NO_ANSWER
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
        self._indent = ""
        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb) if config.vecdb else None
        self.parser: Optional[Parser] = (
            Parser(config.parsing) if config.parsing else None
        )

    @property
    def indent(self) -> str:
        """Indentation to print before any responses from the agent's entities."""
        return self._indent

    @indent.setter
    def indent(self, value: str) -> None:
        self._indent = value

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
        """
        Response from the "agent itself", i.e., from any application "tool method"
        that was triggerred by input_str (if it contained a json substring matching
        a handler method).
        Args:
            input_str (str): the input string to respond to

        Returns:
            Optional[ChatDocument]: the response, packaged as a ChatDocument

        """
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

    def user_response(self, msg: Optional[str] = None) -> Optional[ChatDocument]:
        """
        Get user response to current message. Could allow (human) user to intervene
        with an actual answer, or quit using "q" or "x"

        Args:
            msg (str): the string to respond to.

        Returns:
            (str) User response, packaged as a ChatDocument

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

    def llm_response(self, prompt: Optional[str] = None) -> Optional[ChatDocument]:
        """
        LLM response to a prompt.
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
