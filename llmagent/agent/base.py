from abc import ABC
from typing import Dict, Optional, Type, List
from contextlib import ExitStack
from pydantic import BaseSettings, ValidationError
from halo import Halo
from llmagent.mytypes import Document
from rich import print
import json
from llmagent.agent.message import AgentMessage
from llmagent.language_models.base import LanguageModel, LLMMessage
from llmagent.vector_store.base import VectorStore
from llmagent.parsing.parser import Parser
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
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
    vecdb: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig()


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.dialog = []  # seq of LLM (prompt, response) tuples
        self.response: Document = None  # last response
        self.handled_classes: Dict[str, Type[AgentMessage]] = {}

        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb)
        self.parser = Parser(config.parsing)

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

    def message_instructions(self):
        """
        Generate a string containing instructions to the LLM on when to format
        responses as JSON, based on enabled message classes.

        Returns:
            str: The instructions string.
        """
        enabled_classes: List[Type[AgentMessage]] = self.handled_classes.values()
        instructions = [
            f"JSON CONDITION {i+1}: " + c().usage_instruction()
            for i, c in enumerate(enabled_classes)
        ]
        json_conditions = """
        If your QUESTION fits one of the  below JSON CONDITIONS, then FORMAT the 
        question in the JSON format indicated; otherwise, keep it in the original form.
        """ + "\n\n".join(
            instructions
        )
        conversation_example = self.sample_dialog()

        return f"""
        FORMATTING RULES:
        {json_conditions}
        
        SAMPLE CONVERSATION:
        {conversation_example}
        
        Now start asking me questions. Ignore all specific details above, those were 
        just examples. Start from scratch, assume you know nothing. Remember to 
        format the question in JSON if it fits one of the JSON CONDITIONs above.          
        """

    def sample_dialog(self):
        """
        Generate a sample dialog based on enabled message classes.
        Returns:
            str: The sample dialog string.
        """
        enabled_classes: List[Type[AgentMessage]] = self.handled_classes.values()
        # use at most 2 sample conversations, no need to be exhaustive
        sample_convo = [
            c().sample_conversation() for i, c in enumerate(enabled_classes) if i < 2
        ]
        return "\n\n".join(sample_convo)

    @staticmethod
    def _extract_json(input_str: str) -> Optional[str]:
        """
        Extract the JSON string from the input string.

        Args:
            input_str (str): The input string containing JSON.

        Returns:
            Optional[str]: The JSON string if found, otherwise None.
        """
        try:
            start_index = input_str.index("{")
            end_index = input_str.rindex("}")
            return input_str[start_index : (end_index + 1)]
        except ValueError:
            return None

    def handle_message(self, input_str: str) -> Optional[str]:
        """
        Route the input string to the appropriate handler method based on the
        message class.

        Args:
            input_str (str): The input string containing JSON.

        Returns:
            Optional[Str]: The result of the handler method in string form so it can
            be sent back to the LLM, or None if the input string was not successfully
            by a method.
        """
        json_str = self._extract_json(input_str)
        if json_str is None:
            return None

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

    def run(self):
        while True:
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
