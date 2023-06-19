import json
import logging
from abc import ABC
from contextlib import ExitStack
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type, no_type_check

from pydantic import BaseSettings, ValidationError
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from llmagent.agent.message import INSTRUCTION, AgentMessage
from llmagent.language_models.base import LanguageModel, LLMConfig, LLMFunctionCall
from llmagent.mytypes import DocMetaData, Document
from llmagent.parsing.json import extract_top_level_json
from llmagent.parsing.parser import Parser, ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.configuration import settings
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
    sender_name: str = ""
    recipient: str = ""
    usage: int = 0
    cached: bool = False
    displayed: bool = False


class ChatDocument(Document):
    function_call: Optional[LLMFunctionCall] = None
    metadata: ChatDocMetaData


class AgentConfig(BaseSettings):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components, in a hierarchy. Let us see how this works.
    """

    name: str = "LLM-Agent"
    debug: bool = False
    vecdb: Optional[VectorStoreConfig] = VectorStoreConfig()
    llm: Optional[LLMConfig] = LLMConfig()
    parsing: Optional[ParsingConfig] = ParsingConfig()
    prompts: Optional[PromptsConfig] = PromptsConfig()


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.dialog: List[Tuple[str, str]] = []  # seq of LLM (prompt, response) tuples
        self.llm_tools_map: Dict[str, Type[AgentMessage]] = {}
        self.llm_tools_handled: Set[str] = set()
        self.llm_tools_usable: Set[str] = set()
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

    def _get_tool_list(
        self, message_class: Optional[Type[AgentMessage]] = None
    ) -> List[str]:
        if message_class is None:
            return list(self.llm_tools_map.keys())
        else:
            if not issubclass(message_class, AgentMessage):
                raise ValueError("message_class must be a subclass of AgentMessage")
            tool = message_class.default_value("request")
            self.llm_tools_map[tool] = message_class
            return [tool]

    def enable_message_handling(
        self, message_class: Optional[Type[AgentMessage]] = None
    ) -> None:
        """
        Enable an agent to RESPOND (i.e. handle) a "tool" message of a specific type
            from LLM. Also "registers" (i.e. adds) the `message_class` to the
            `self.llm_tools_map` dict.

        Args:
            message_class (Optional[Type[AgentMessage]]): The message class to enable;
                Optional; if None, all known message classes are enabled for handling.

        """
        for t in self._get_tool_list(message_class):
            self.llm_tools_handled.add(t)

    def disable_message_handling(
        self,
        message_class: Optional[Type[AgentMessage]] = None,
    ) -> None:
        """
        Disable a message class from being handled by this Agent.

        Args:
            message_class (Optional[Type[AgentMessage]]): The message class to disable.
                If None, all message classes are disabled.
        """
        for t in self._get_tool_list(message_class):
            self.llm_tools_handled.discard(t)

    def json_format_rules(self) -> str:
        """
        Specification of JSON formatting rules, based on the currently enabled
        message classes.

        Returns:
            str: formatting rules
        """
        enabled_classes: List[Type[AgentMessage]] = list(self.llm_tools_map.values())
        if len(enabled_classes) == 0:
            return "You can ask questions in natural language."

        json_conditions = "\n\n".join(
            [
                str(msg_cls.default_value("request"))
                + ":\n"
                + str(msg_cls.default_value("purpose"))
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
        enabled_classes: List[Type[AgentMessage]] = list(self.llm_tools_map.values())
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

    def agent_response(
        self, input_str: Optional[str] = None, sender_name: str = ""
    ) -> Optional[ChatDocument]:
        """
        Response from the "agent itself", i.e., from any application "tool method"
        that was triggerred by input_str (if it contained a json substring matching
        a handler method).
        Args:
            input_str (str): the input string to respond to
            sender_name (str): the name of the sender (ignored for now, but including
                it here so all *_response methods have the same signature)

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
                sender_name=self.config.name,
            ),
        )

    def user_response(
        self, msg: Optional[str] = None, sender_name: str = ""
    ) -> Optional[ChatDocument]:
        """
        Get user response to current message. Could allow (human) user to intervene
        with an actual answer, or quit using "q" or "x"

        Args:
            msg (str): the string to respond to.
            sender_name (str): the name of the sender (ignored for now, but including
                it here so all *_response methods have the same signature)

        Returns:
            (str) User response, packaged as a ChatDocument

        """
        if self.default_human_response is not None:
            # useful for automated testing
            user_msg = self.default_human_response
        elif not settings.interactive:
            user_msg = ""
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

    @no_type_check
    def llm_response(
        self, prompt: Optional[str] = None, sender_name: str = ""
    ) -> Optional[ChatDocument]:
        """
        LLM response to a prompt.
        Args:
            prompt (str): prompt string
            sender_name (str): the name of the sender (ignored for completion mode,
                but used in chat_completion in `chat_agent.py`)

        Returns:
            Response from LLM, packaged as a ChatDocument
        """
        if prompt is None or self.llm is None:
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
            function_call=response.function_call,
            metadata=DocMetaData(
                source=Entity.LLM.value,
                sender=Entity.LLM.value,
                usage=response.usage,
                displayed=displayed,
                cached=response.cached,
            ),
        )

    def get_tool_messages(self, msg: str | ChatDocument) -> List[AgentMessage]:
        if isinstance(msg, str):
            return self.get_json_tool_messages(msg)
        assert isinstance(msg, ChatDocument)
        if msg.content != "":
            return self.get_json_tool_messages(msg.content)

        if msg.function_call is None:
            return []
        tool_name = msg.function_call.name
        tool_msg = msg.function_call.arguments or {}
        if tool_name not in self.llm_tools_handled:
            return []
        tool_class = self.llm_tools_map[tool_name]
        tool_msg.update(dict(request=tool_name))
        try:
            tool = tool_class.parse_obj(tool_msg)
        except ValidationError as ve:
            raise ValueError("Error parsing tool_msg as message class") from ve
        return [tool]

    def get_json_tool_messages(self, input_str: str) -> List[AgentMessage]:
        """
        Returns AgentMessage objects (tools) corresponding to JSON substrings, if any.

        Args:
            input_str (str): input string, typically a message sent by an LLM

        Returns:
            List[AgentMessage]: list of AgentMessage objects
        """
        json_substrings = extract_top_level_json(input_str)
        if len(json_substrings) == 0:
            return []
        results = [self._get_one_tool_message(j) for j in json_substrings]
        return [r for r in results if r is not None]

    def handle_message(self, msg: str | ChatDocument) -> Optional[str]:
        """
        Handle a "tool" message either a string containing one or more
        valie "tool" JSON substrings,  or a
        ChatDocument containing a `function_call` attribute.
        Handle with the corresponding handler method, and return
        the results as a combined string.

        Args:
            msg (str | ChatDocument): The string or ChatDocument to handle

        Returns:
            Optional[Str]: The result of the handler method in string form so it can
            be sent back to the LLM, or None if `msg` was not successfully
            handled by a method.
        """
        tools = self.get_tool_messages(msg)
        if len(tools) == 0:
            return self.handle_message_fallback(msg)

        results = [self.handle_tool_message(t) for t in tools]

        results_list = [r for r in results if r is not None]
        if len(results_list) == 0:
            return self.handle_message_fallback(msg)
        # there was a non-None result
        final = "\n".join(results_list)
        assert (
            final != ""
        ), """final result from a handler should not be empty str, since that would be 
            considered an invalid result and other responders will be tried, 
            and we may not necessarily want that"""
        return final

    def handle_message_fallback(self, msg: str | ChatDocument) -> Optional[str]:
        """
        Fallback method to handle possible "tool" msg if not other method applies
        or if an error is thrown.
        This method can be overridden by subclasses.

        Args:
            msg (str | ChatDocument): The input msg to handle
        Returns:
            str: The result of the handler method in string form so it can
                be sent back to the LLM.
        """
        return None

    def _get_one_tool_message(self, json_str: str) -> Optional[AgentMessage]:
        json_data = json.loads(json_str)
        request = json_data.get("request")
        if request is None or request not in self.llm_tools_handled:
            return None

        message_class = self.llm_tools_map.get(request)
        if message_class is None:
            logger.warning(f"No message class found for request '{request}'")
            return None

        try:
            message = message_class.parse_obj(json_data)
        except ValidationError as ve:
            raise ValueError("Error parsing JSON as message class") from ve
        return message

    def handle_tool_message(self, tool: AgentMessage) -> Optional[str]:
        """
        Respond to a tool request from the LLM, in the form of an AgentMessage object.
        Args:
            tool: AgentMessage object representing the tool request.

        Returns:

        """
        tool_name = tool.default_value("request")
        handler_method = getattr(self, tool_name, None)
        if handler_method is None:
            return None

        return handler_method(tool)  # type: ignore

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
