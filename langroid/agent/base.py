import json
import logging
from abc import ABC
from contextlib import ExitStack
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, cast, no_type_check

from pydantic import BaseSettings, ValidationError
from rich import print
from rich.console import Console
from rich.prompt import Prompt

from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.tool_message import INSTRUCTION, ToolMessage
from langroid.language_models.base import (
    LanguageModel,
    LLMConfig,
)
from langroid.mytypes import DocMetaData, Entity
from langroid.parsing.json import extract_top_level_json
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import settings
from langroid.utils.constants import NO_ANSWER
from langroid.vector_store.base import VectorStore, VectorStoreConfig

console = Console()

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """
    General config settings for an LLM agent. This is nested, combining configs of
    various components.
    """

    name: str = "LLM-Agent"
    debug: bool = False
    vecdb: Optional[VectorStoreConfig] = VectorStoreConfig()
    llm: Optional[LLMConfig] = LLMConfig()
    parsing: Optional[ParsingConfig] = ParsingConfig()
    prompts: Optional[PromptsConfig] = PromptsConfig()


class Agent(ABC):
    """
    An Agent is an abstraction that encapsulates mainly two components:

    - a language model (LLM)
    - a vector store (vecdb)

    plus associated components such as a parser, and variables that hold
    information about any tool/function-calling messages that have been defined.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.dialog: List[Tuple[str, str]] = []  # seq of LLM (prompt, response) tuples
        self.llm_tools_map: Dict[str, Type[ToolMessage]] = {}
        self.llm_tools_handled: Set[str] = set()
        self.llm_tools_usable: Set[str] = set()
        self.default_human_response: Optional[str] = None
        self._indent = ""
        self.llm = LanguageModel.create(config.llm)
        self.vecdb = VectorStore.create(config.vecdb) if config.vecdb else None
        self.parser: Optional[Parser] = (
            Parser(config.parsing) if config.parsing else None
        )

    def entity_responders(
        self,
    ) -> List[
        Tuple[Entity, Callable[[None | str | ChatDocument], None | ChatDocument]]
    ]:
        """
        Sequence of (entity, response_method) pairs. This sequence is used
            in a `Task` to respond to the current pending message.
            See `Task.step()` for details.
        Returns:
            Sequence of (entity, response_method) pairs.
        """
        return [
            (Entity.AGENT, self.agent_response),
            (Entity.LLM, self.llm_response),
            (Entity.USER, self.user_response),
        ]

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
        self, message_class: Optional[Type[ToolMessage]] = None
    ) -> List[str]:
        if message_class is None:
            return list(self.llm_tools_map.keys())
        else:
            if not issubclass(message_class, ToolMessage):
                raise ValueError("message_class must be a subclass of ToolMessage")
            tool = message_class.default_value("request")
            self.llm_tools_map[tool] = message_class
            return [tool]

    def enable_message_handling(
        self, message_class: Optional[Type[ToolMessage]] = None
    ) -> None:
        """
        Enable an agent to RESPOND (i.e. handle) a "tool" message of a specific type
            from LLM. Also "registers" (i.e. adds) the `message_class` to the
            `self.llm_tools_map` dict.

        Args:
            message_class (Optional[Type[ToolMessage]]): The message class to enable;
                Optional; if None, all known message classes are enabled for handling.

        """
        for t in self._get_tool_list(message_class):
            self.llm_tools_handled.add(t)

    def disable_message_handling(
        self,
        message_class: Optional[Type[ToolMessage]] = None,
    ) -> None:
        """
        Disable a message class from being handled by this Agent.

        Args:
            message_class (Optional[Type[ToolMessage]]): The message class to disable.
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
        enabled_classes: List[Type[ToolMessage]] = list(self.llm_tools_map.values())
        if len(enabled_classes) == 0:
            return "You can ask questions in natural language."

        json_conditions = "\n\n".join(
            [
                str(msg_cls.default_value("request"))
                + ":\n"
                + str(msg_cls.default_value("purpose"))
                for i, msg_cls in enumerate(enabled_classes)
                if msg_cls.default_value("request") in self.llm_tools_usable
            ]
        )
        return json_conditions

    def sample_multi_round_dialog(self) -> str:
        """
        Generate a sample multi-round dialog based on enabled message classes.
        Returns:
            str: The sample dialog string.
        """
        enabled_classes: List[Type[ToolMessage]] = list(self.llm_tools_map.values())
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
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        Response from the "agent itself" handling a "tool message"
        or LLM's `function_call` (e.g. OpenAI `function_call`)
        Args:
            msg (str|ChatDocument): the input to respond to: if msg is a string,
                and it contains a valid JSON-structured "tool message", or
                if msg is a ChatDocument, and it contains a `function_call`.
        Returns:
            Optional[ChatDocument]: the response, packaged as a ChatDocument

        """
        if msg is None:
            return None

        results = self.handle_message(msg)
        if results is None:
            return None
        console.print(f"[red]{self.indent}", end="")
        print(f"[red]Agent: {results}")
        sender_name = self.config.name
        if isinstance(msg, ChatDocument) and msg.function_call is not None:
            # if result was from handling an LLM `function_call`,
            # set sender_name to "request", i.e. name of the function_call
            sender_name = msg.function_call.name

        return ChatDocument(
            content=results,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                sender_name=sender_name,
            ),
        )

    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        Get user response to current message. Could allow (human) user to intervene
        with an actual answer, or quit using "q" or "x"

        Args:
            msg (str|ChatDocument): the string to respond to.

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
                    source=Entity.USER,
                    sender=Entity.USER,
                ),
            )

    @no_type_check
    def llm_can_respond(self, message: Optional[str | ChatDocument] = None) -> bool:
        """
        Whether the LLM can respond to a message.
        Args:
            message (str|ChatDocument): message or ChatDocument object to respond to.

        Returns:

        """
        if self.llm is None:
            return False

        if isinstance(message, ChatDocument) and message.function_call is not None:
            # LLM should not handle `function_call` messages,
            # EVEN if message.function_call is not a legit function_call
            # The OpenAI API raises error if there is a message in history
            # from a non-Assistant role, with a `function_call` in it
            return False

        if message is not None and len(self.get_tool_messages(message)) > 0:
            # if there is a valid "tool" message (either JSON or via `function_call`)
            # then LLM cannot respond to it
            return False

        return True

    @no_type_check
    def llm_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        LLM response to a prompt.
        Args:
            msg (str|ChatDocument): prompt string, or ChatDocument object

        Returns:
            Response from LLM, packaged as a ChatDocument
        """
        if msg is None or not self.llm_can_respond(msg):
            return None

        if isinstance(msg, ChatDocument):
            prompt = msg.content
        else:
            prompt = msg

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

        return ChatDocument.from_LLMResponse(response, displayed)

    def get_tool_messages(self, msg: str | ChatDocument) -> List[ToolMessage]:
        if isinstance(msg, str):
            return self.get_json_tool_messages(msg)
        assert isinstance(msg, ChatDocument)
        # when `content` is non-empty, we assume there will be no `function_call`
        if msg.content != "":
            return self.get_json_tool_messages(msg.content)

        # otherwise, we check look for a `function_call`
        fun_call_cls = self.get_function_call_class(msg)
        return [fun_call_cls] if fun_call_cls is not None else []

    def get_json_tool_messages(self, input_str: str) -> List[ToolMessage]:
        """
        Returns ToolMessage objects (tools) corresponding to JSON substrings, if any.

        Args:
            input_str (str): input string, typically a message sent by an LLM

        Returns:
            List[ToolMessage]: list of ToolMessage objects
        """
        json_substrings = extract_top_level_json(input_str)
        if len(json_substrings) == 0:
            return []
        results = [self._get_one_tool_message(j) for j in json_substrings]
        return [r for r in results if r is not None]

    def get_function_call_class(self, msg: ChatDocument) -> Optional[ToolMessage]:
        if msg.function_call is None:
            return None
        tool_name = msg.function_call.name
        tool_msg = msg.function_call.arguments or {}
        if tool_name not in self.llm_tools_handled:
            raise ValueError(f"{tool_name} is not a valid function_call!")
        tool_class = self.llm_tools_map[tool_name]
        tool_msg.update(dict(request=tool_name))
        try:
            tool = tool_class.parse_obj(tool_msg)
        except ValidationError as ve:
            raise ValueError("Error parsing tool_msg as message class") from ve
        return tool

    def tool_validation_error(self, ve: ValidationError) -> str:
        """
        Handle a validation error raised when parsing a tool message,
            when there is a legit tool name used, but it has missing/bad fields.
        Args:
            tool (ToolMessage): The tool message that failed validation
            ve (ValidationError): The exception raised

        Returns:
            str: The error message to send back to the LLM
        """
        tool_name = cast(ToolMessage, ve.model).default_value("request")
        bad_field_errors = "\n".join(
            [f"{e['loc'][0]}: {e['msg']}" for e in ve.errors() if "loc" in e]
        )
        return f"""
        There were one or more errors in your attempt to use the 
        TOOL or function_call named '{tool_name}': 
        {bad_field_errors}
        Please write your message again, correcting the errors.
        """

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
        try:
            tools = self.get_tool_messages(msg)
        except ValidationError as ve:
            # correct tool name but bad fields
            return self.tool_validation_error(ve)
        except ValueError:
            # invalid tool name
            # We return None since returning "invalid tool name" would
            # be considered a valid result in task loop, and would be treated
            # as a response to the tool message even though the tool was not intended
            # for this agent.
            return None
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

    def _get_one_tool_message(self, json_str: str) -> Optional[ToolMessage]:
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
            raise ve
        return message

    def handle_tool_message(self, tool: ToolMessage) -> Optional[str]:
        """
        Respond to a tool request from the LLM, in the form of an ToolMessage object.
        Args:
            tool: ToolMessage object representing the tool request.

        Returns:

        """
        tool_name = tool.default_value("request")
        handler_method = getattr(self, tool_name, None)
        if handler_method is None:
            return None

        try:
            result = handler_method(tool)
        except Exception as e:
            logger.warning(f"Error handling tool-message {tool_name}: {e}")
            return None
        return result  # type: ignore

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
        This is not currently used, since we rely on the task loop and "TO:" syntax
        to send requests to other agents. It is generally best to avoid using this
        method.

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
