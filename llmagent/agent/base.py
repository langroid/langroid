from abc import ABC
from typing import Dict, Optional, Type, List
from contextlib import ExitStack
from pydantic import BaseSettings, ValidationError
from halo import Halo
from llmagent.mytypes import Document
from rich import print
import json
from llmagent.agent.message import AgentMessage, INSTRUCTION
from llmagent.language_models.base import LanguageModel, LLMMessage
from llmagent.vector_store.base import VectorStore
from llmagent.parsing.parser import Parser
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.parsing.json import extract_top_level_json
from llmagent.prompts.prompts_config import PromptsConfig
import logging

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """

    name: str = "llmagent"
    debug: bool = False
    stream: bool = False  # stream LLM output?
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

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb) if config.vecdb else None
        self.parser = Parser(config.parsing) if config.parsing else None

    def update_dialog(self, prompt, output):
        self.dialog.append((prompt, output))

    def get_dialog(self):
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
        # include non-JSON sample only for the first message class
        sample_convo = [
            msg_cls().sample_conversation(json_only=(i == 0))
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
        json_str = json_str.replace("'", '"')
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

    def run(self, iters: int = -1) -> None:
        """
        Run the LLM in interactive mode, asking for input and generating responses.
        If iters > 0, quit after that many iterations.
        Args:
            iters: number of iterations to run, if > 0
        """
        niters = 0
        while True:
            if iters > 0 and niters >= iters:
                break
            niters += 1
            print("\n[blue]Query: ", end="")
            query = input("")
            if query in ["exit", "quit", "q", "x", "bye"]:
                print("[green] Bye, it has been a pleasure, hope this was useful!")
                break
            self.respond(query)

    def respond(self, query: str) -> Document:
        """
        Respond to a query.
        Args:
            query: query string
        Returns:
            Document
        """

        with ExitStack() as stack:  # for conditionally using Halo spinner
            if not self.llm.get_stream():
                # show Halo spinner only if not streaming!
                cm = Halo(text="LLM responding to message...", spinner="dots")
                stack.enter_context(cm)
            response = self.llm.generate(query, self.config.llm.max_tokens)
        if not self.llm.get_stream():
            print("[green]" + response.message)

        return Document(
            content=response.message, metadata=dict(source="LLM", usage=response.usage)
        )

    def respond_messages(self, messages: List[LLMMessage]) -> Document:
        """
        Respond to a series of messages, e.g. with OpenAI ChatCompletion
        Args:
            messages: seq of messages (with role, content fields) sent to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        with ExitStack() as stack:  # for conditionally using Halo spinner
            if not self.llm.get_stream():
                # show Halo spinner only if not streaming!
                cm = Halo(text="LLM responding to messages...", spinner="dots")
                stack.enter_context(cm)
            response = self.llm.chat(messages, self.config.llm.max_tokens)
        if not self.llm.get_stream():
            print("[green]" + response.message)
        return Document(
            content=response.message, metadata=dict(source="LLM", usage=response.usage)
        )
