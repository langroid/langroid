import copy
import inspect
import logging
import textwrap
from contextlib import ExitStack
from typing import Dict, List, Optional, Set, Tuple, Type, cast

from rich import print
from rich.console import Console
from rich.markup import escape

from langroid.agent.base import Agent, AgentConfig, async_noop_fn, noop_fn
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.language_models.base import (
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    OpenAIToolSpec,
    Role,
    StreamingIfAllowed,
    ToolChoiceTypes,
)
from langroid.language_models.openai_gpt import OpenAIGPT
from langroid.utils.configuration import settings
from langroid.utils.object_registry import ObjectRegistry
from langroid.utils.output import status

console = Console()

logger = logging.getLogger(__name__)


class ChatAgentConfig(AgentConfig):
    """
    Configuration for ChatAgent
    Attributes:
        system_message: system message to include in message sequence
             (typically defines role and task of agent).
             Used only if `task` is not specified in the constructor.
        user_message: user message to include in message sequence.
             Used only if `task` is not specified in the constructor.
        use_tools: whether to use our own ToolMessages mechanism
        use_functions_api: whether to use functions/tools native to the LLM API
                (e.g. OpenAI's `function_call` or `tool_call` mechanism)
        use_tools_api: When `use_functions_api` is True, if this is also True,
            the OpenAI tool-call API is used, rather than the older/deprecated
            function-call API. However the tool-call API has some tricky aspects,
            hence we set this to False by default.
        enable_orchestration_tool_handling: whether to enable handling of orchestration
            tools, e.g. ForwardTool, DoneTool, PassTool, etc.
    """

    system_message: str = "You are a helpful assistant."
    user_message: Optional[str] = None
    use_tools: bool = False
    use_functions_api: bool = True
    use_tools_api: bool = False
    enable_orchestration_tool_handling: bool = True

    def _set_fn_or_tools(self, fn_available: bool) -> None:
        """
        Enable Langroid Tool or OpenAI-like fn-calling,
        depending on config settings and availability of fn-calling.
        """
        if self.use_functions_api and not fn_available:
            logger.debug(
                """
                You have enabled `use_functions_api` but the LLM does not support it.
                So we will enable `use_tools` instead, so we can use 
                Langroid's ToolMessage mechanism.
                """
            )
            self.use_functions_api = False
            self.use_tools = True

        if not self.use_functions_api or not self.use_tools:
            return
        if self.use_functions_api and self.use_tools:
            logger.debug(
                """
                You have enabled both `use_tools` and `use_functions_api`.
                Turning off `use_tools`, since the LLM supports function-calling.
                """
            )
            self.use_tools = False
            self.use_functions_api = True


class ChatAgent(Agent):
    """
    Chat Agent interacting with external env
    (could be human, or external tools).
    The agent (the LLM actually) is provided with an optional "Task Spec",
    which is a sequence of `LLMMessage`s. These are used to initialize
    the `task_messages` of the agent.
    In most applications we will use a `ChatAgent` rather than a bare `Agent`.
    The `Agent` class mainly exists to hold various common methods and attributes.
    One difference between `ChatAgent` and `Agent` is that `ChatAgent`'s
    `llm_response` method uses "chat mode" API (i.e. one that takes a
    message sequence rather than a single message),
    whereas the same method in the `Agent` class uses "completion mode" API (i.e. one
    that takes a single message).
    """

    def __init__(
        self,
        config: ChatAgentConfig = ChatAgentConfig(),
        task: Optional[List[LLMMessage]] = None,
    ):
        """
        Chat-mode agent initialized with task spec as the initial message sequence
        Args:
            config: settings for the agent

        """
        super().__init__(config)
        self.config: ChatAgentConfig = config
        self.config._set_fn_or_tools(self._fn_call_available())
        self.message_history: List[LLMMessage] = []
        self.init_state()
        # An agent's "task" is defined by a system msg and an optional user msg;
        # These are "priming" messages that kick off the agent's conversation.
        self.system_message: str = self.config.system_message
        self.user_message: str | None = self.config.user_message

        if task is not None:
            # if task contains a system msg, we override the config system msg
            if len(task) > 0 and task[0].role == Role.SYSTEM:
                self.system_message = task[0].content
            # if task contains a user msg, we override the config user msg
            if len(task) > 1 and task[1].role == Role.USER:
                self.user_message = task[1].content

        # system-level instructions for using tools/functions:
        # We maintain these as tools/functions are enabled/disabled,
        # and whenever an LLM response is sought, these are used to
        # recreate the system message (via `_create_system_and_tools_message`)
        # each time, so it reflects the current set of enabled tools/functions.
        # (a) these are general instructions on using certain tools/functions,
        #   if they are specified in a ToolMessage class as a classmethod `instructions`
        self.system_tool_instructions: str = ""
        # (b) these are only for the builtin in Langroid TOOLS mechanism:
        self.system_tool_format_instructions: str = ""

        self.llm_functions_map: Dict[str, LLMFunctionSpec] = {}
        self.llm_functions_handled: Set[str] = set()
        self.llm_functions_usable: Set[str] = set()
        self.llm_function_force: Optional[Dict[str, str]] = None

        if self.config.enable_orchestration_tool_handling:
            # Only enable HANDLING by `agent_response`, NOT LLM generation of these.
            # This is useful where tool-handlers or agent_response generate these
            # tools, and need to be handled.
            # We don't want enable orch tool GENERATION by default, since that
            # might clutter-up the LLM system message unnecessarily.
            from langroid.agent.tools.orchestration import (
                AgentDoneTool,
                AgentSendTool,
                DonePassTool,
                DoneTool,
                ForwardTool,
                PassTool,
                ResultTool,
                SendTool,
            )

            self.enable_message(ForwardTool, use=False, handle=True)
            self.enable_message(DoneTool, use=False, handle=True)
            self.enable_message(AgentDoneTool, use=False, handle=True)
            self.enable_message(PassTool, use=False, handle=True)
            self.enable_message(DonePassTool, use=False, handle=True)
            self.enable_message(SendTool, use=False, handle=True)
            self.enable_message(AgentSendTool, use=False, handle=True)
            self.enable_message(ResultTool, use=False, handle=True)

    def init_state(self) -> None:
        """
        Initialize the state of the agent. Just conversation state here,
        but subclasses can override this to initialize other state.
        """
        super().init_state()
        self.clear_history(0)
        self.clear_dialog()

    @staticmethod
    def from_id(id: str) -> "ChatAgent":
        """
        Get an agent from its ID
        Args:
            agent_id (str): ID of the agent
        Returns:
            ChatAgent: The agent with the given ID
        """
        return cast(ChatAgent, Agent.from_id(id))

    def clone(self, i: int = 0) -> "ChatAgent":
        """Create i'th clone of this agent, ensuring tool use/handling is cloned.
        Important: We assume all member variables are in the __init__ method here
        and in the Agent class.
        TODO: We are attempting to clone an agent after its state has been
        changed in possibly many ways. Below is an imperfect solution. Caution advised.
        Revisit later.
        """
        agent_cls = type(self)
        config_copy = copy.deepcopy(self.config)
        config_copy.name = f"{config_copy.name}-{i}"
        new_agent = agent_cls(config_copy)
        new_agent.system_tool_instructions = self.system_tool_instructions
        new_agent.system_tool_format_instructions = self.system_tool_format_instructions
        new_agent.llm_tools_map = self.llm_tools_map
        new_agent.llm_functions_map = self.llm_functions_map
        new_agent.llm_functions_handled = self.llm_functions_handled
        new_agent.llm_functions_usable = self.llm_functions_usable
        new_agent.llm_function_force = self.llm_function_force
        # Caution - we are copying the vector-db, maybe we don't always want this?
        new_agent.vecdb = self.vecdb
        new_agent.id = ObjectRegistry.new_id()
        if self.config.add_to_registry:
            ObjectRegistry.register_object(new_agent)
        return new_agent

    def _fn_call_available(self) -> bool:
        """Does this agent's LLM support function calling?"""
        return (
            self.llm is not None
            and isinstance(self.llm, OpenAIGPT)
            and self.llm.is_openai_chat_model()
            and self.llm.supports_functions_or_tools()
        )

    def set_system_message(self, msg: str) -> None:
        self.system_message = msg
        if len(self.message_history) > 0:
            # if there is message history, update the system message in it
            self.message_history[0].content = msg

    def set_user_message(self, msg: str) -> None:
        self.user_message = msg

    @property
    def task_messages(self) -> List[LLMMessage]:
        """
        The task messages are the initial messages that define the task
        of the agent. There will be at least a system message plus possibly a user msg.
        Returns:
            List[LLMMessage]: the task messages
        """
        msgs = [self._create_system_and_tools_message()]
        if self.user_message:
            msgs.append(LLMMessage(role=Role.USER, content=self.user_message))
        return msgs

    def _drop_msg_update_tool_calls(self, msg: LLMMessage) -> None:
        id2idx = {t.id: i for i, t in enumerate(self.oai_tool_calls)}
        if msg.role == Role.TOOL:
            # dropping tool result, so ADD the corresponding tool-call back
            # to the list of pending calls!
            id = msg.tool_call_id
            if id in self.oai_tool_id2call:
                self.oai_tool_calls.append(self.oai_tool_id2call[id])
        elif msg.tool_calls is not None:
            # dropping a msg with tool-calls, so DROP these from pending list
            # as well as from id -> call map
            for tool_call in msg.tool_calls:
                if tool_call.id in id2idx:
                    self.oai_tool_calls.pop(id2idx[tool_call.id])
                if tool_call.id in self.oai_tool_id2call:
                    del self.oai_tool_id2call[tool_call.id]

    def clear_history(self, start: int = -2) -> None:
        """
        Clear the message history, starting at the index `start`

        Args:
            start (int): index of first message to delete; default = -2
                    (i.e. delete last 2 messages, typically these
                    are the last user and assistant messages)
        """
        if start < 0:
            n = len(self.message_history)
            start = max(0, n + start)
        dropped = self.message_history[start:]
        # consider the dropped msgs in REVERSE order, so we are
        # carefully updating self.oai_tool_calls
        for msg in reversed(dropped):
            self._drop_msg_update_tool_calls(msg)
            # clear out the chat document from the ObjectRegistry
            ChatDocument.delete_id(msg.chat_document_id)
        self.message_history = self.message_history[:start]

    def update_history(self, message: str, response: str) -> None:
        """
        Update the message history with the latest user message and LLM response.
        Args:
            message (str): user message
            response: (str): LLM response
        """
        self.message_history.extend(
            [
                LLMMessage(role=Role.USER, content=message),
                LLMMessage(role=Role.ASSISTANT, content=response),
            ]
        )

    def tool_format_rules(self) -> str:
        """
        Specification of tool formatting rules
        (typically JSON-based but can be non-JSON, e.g. XMLToolMessage),
        based on the currently enabled usable `ToolMessage`s

        Returns:
            str: formatting rules
        """
        # ONLY Usable tools (i.e. LLM-generation allowed),
        usable_tool_classes: List[Type[ToolMessage]] = [
            t
            for t in list(self.llm_tools_map.values())
            if t.default_value("request") in self.llm_tools_usable
        ]

        if len(usable_tool_classes) == 0:
            return "You can ask questions in natural language."
        format_instructions = "\n\n".join(
            [
                msg_cls.format_instructions(tool=self.config.use_tools)
                for msg_cls in usable_tool_classes
            ]
        )
        # if any of the enabled classes has json_group_instructions, then use that,
        # else fall back to ToolMessage.json_group_instructions
        for msg_cls in usable_tool_classes:
            if hasattr(msg_cls, "json_group_instructions") and callable(
                getattr(msg_cls, "json_group_instructions")
            ):
                return msg_cls.group_format_instructions().format(
                    format_instructions=format_instructions
                )
        return ToolMessage.group_format_instructions().format(
            format_instructions=format_instructions
        )

    def tool_instructions(self) -> str:
        """
        Instructions for tools or function-calls, for enabled and usable Tools.
        These are inserted into system prompt regardless of whether we are using
        our own ToolMessage mechanism or the LLM's function-call mechanism.

        Returns:
            str: concatenation of instructions for all usable tools
        """
        enabled_classes: List[Type[ToolMessage]] = list(self.llm_tools_map.values())
        if len(enabled_classes) == 0:
            return ""
        instructions = []
        for msg_cls in enabled_classes:
            if (
                hasattr(msg_cls, "instructions")
                and inspect.ismethod(msg_cls.instructions)
                and msg_cls.default_value("request") in self.llm_tools_usable
            ):
                # example will be shown in tool_format_rules() when using TOOLs,
                # so we don't need to show it here.
                example = "" if self.config.use_tools else (msg_cls.usage_examples())
                if example != "":
                    example = "EXAMPLES:\n" + example
                class_instructions = msg_cls.instructions()
                guidance = (
                    ""
                    if class_instructions == ""
                    else ("GUIDANCE: " + class_instructions)
                )
                if guidance == "" and example == "":
                    continue
                instructions.append(
                    textwrap.dedent(
                        f"""
                        TOOL: {msg_cls.default_value("request")}:
                        {guidance}
                        {example}
                        """.lstrip()
                    )
                )
        if len(instructions) == 0:
            return ""
        instructions_str = "\n\n".join(instructions)
        return textwrap.dedent(
            f"""
            === GUIDELINES ON SOME TOOLS/FUNCTIONS USAGE ===
            {instructions_str}
            """.lstrip()
        )

    def augment_system_message(self, message: str) -> None:
        """
        Augment the system message with the given message.
        Args:
            message (str): system message
        """
        self.system_message += "\n\n" + message

    def last_message_with_role(self, role: Role) -> LLMMessage | None:
        """from `message_history`, return the last message with role `role`"""
        n_role_msgs = len([m for m in self.message_history if m.role == role])
        if n_role_msgs == 0:
            return None
        idx = self.nth_message_idx_with_role(role, n_role_msgs)
        return self.message_history[idx]

    def nth_message_idx_with_role(self, role: Role, n: int) -> int:
        """Index of `n`th message in message_history, with specified role.
        (n is assumed to be 1-based, i.e. 1 is the first message with that role).
        Return -1 if not found. Index = 0 is the first message in the history.
        """
        indices_with_role = [
            i for i, m in enumerate(self.message_history) if m.role == role
        ]

        if len(indices_with_role) < n:
            return -1
        return indices_with_role[n - 1]

    def update_last_message(self, message: str, role: str = Role.USER) -> None:
        """
        Update the last message that has role `role` in the message history.
        Useful when we want to replace a long user prompt, that may contain context
        documents plus a question, with just the question.
        Args:
            message (str): new message to replace with
            role (str): role of message to replace
        """
        if len(self.message_history) == 0:
            return
        # find last message in self.message_history with role `role`
        for i in range(len(self.message_history) - 1, -1, -1):
            if self.message_history[i].role == role:
                self.message_history[i].content = message
                break

    def _create_system_and_tools_message(self) -> LLMMessage:
        """
        (Re-)Create the system message for the LLM of the agent,
        taking into account any tool instructions that have been added
        after the agent was initialized.

        The system message will consist of:
        (a) the system message from the `task` arg in constructor, if any,
            otherwise the default system message from the config
        (b) the system tool instructions, if any
        (c) the system json tool instructions, if any

        Returns:
            LLMMessage object
        """
        content = textwrap.dedent(
            f"""
            {self.system_message}
            
            {self.system_tool_instructions}
            
            {self.system_tool_format_instructions}
            
            """.lstrip()
        )
        # remove leading and trailing newlines and other whitespace
        return LLMMessage(role=Role.SYSTEM, content=content.strip())

    def unhandled_tools(self) -> set[str]:
        """The set of tools that are known but not handled.
        Useful in task flow: an agent can refuse to accept an incoming msg
        when it only has unhandled tools.
        """
        return self.llm_tools_known - self.llm_tools_handled

    def enable_message(
        self,
        message_class: Optional[Type[ToolMessage] | List[Type[ToolMessage]]],
        use: bool = True,
        handle: bool = True,
        force: bool = False,
        require_recipient: bool = False,
        include_defaults: bool = True,
    ) -> None:
        """
        Add the tool (message class) to the agent, and enable either
        - tool USE (i.e. the LLM can generate JSON to use this tool),
        - tool HANDLING (i.e. the agent can handle JSON from this tool),

        Args:
            message_class: The ToolMessage class OR List of such classes to enable,
                for USE, or HANDLING, or both.
                If this is a list of ToolMessage classes, then the remain args are
                applied to all classes.
                Optional; if None, then apply the enabling to all tools in the
                agent's toolset that have been enabled so far.
            use: IF True, allow the agent (LLM) to use this tool (or all tools),
                else disallow
            handle: if True, allow the agent (LLM) to handle (i.e. respond to) this
                tool (or all tools)
            force: whether to FORCE the agent (LLM) to USE the specific
                 tool represented by `message_class`.
                 `force` is ignored if `message_class` is None.
            require_recipient: whether to require that recipient be specified
                when using the tool message (only applies if `use` is True).
            include_defaults: whether to include fields that have default values,
                in the "properties" section of the JSON format instructions.
                (Normally the OpenAI completion API ignores these fields,
                but the Assistant fn-calling seems to pay attn to these,
                and if we don't want this, we should set this to False.)
        """
        if message_class is not None and isinstance(message_class, list):
            for mc in message_class:
                self.enable_message(
                    mc,
                    use=use,
                    handle=handle,
                    force=force,
                    require_recipient=require_recipient,
                    include_defaults=include_defaults,
                )
            return None
        if require_recipient and message_class is not None:
            message_class = message_class.require_recipient()
        if isinstance(message_class, XMLToolMessage):
            # XMLToolMessage is not compatible with OpenAI's Tools/functions API,
            # so we disable use of functions API, enable langroid-native Tools,
            # which are prompt-based.
            self.config.use_functions_api = False
            self.config.use_tools = True
        super().enable_message_handling(message_class)  # enables handling only
        tools = self._get_tool_list(message_class)
        if message_class is not None:
            request = message_class.default_value("request")
            if request == "":
                raise ValueError(
                    f"""
                    ToolMessage class {message_class} must have a non-empty 
                    'request' field if it is to be enabled as a tool.
                    """
                )
            llm_function = message_class.llm_function_schema(defaults=include_defaults)
            self.llm_functions_map[request] = llm_function
            if force:
                self.llm_function_force = dict(name=request)
            else:
                self.llm_function_force = None

        for t in tools:
            self.llm_tools_known.add(t)

            if handle:
                self.llm_tools_handled.add(t)
                self.llm_functions_handled.add(t)
            else:
                self.llm_tools_handled.discard(t)
                self.llm_functions_handled.discard(t)

            if use:
                tool_class = self.llm_tools_map[t]
                if tool_class._allow_llm_use:
                    self.llm_tools_usable.add(t)
                    self.llm_functions_usable.add(t)
                else:
                    logger.warning(
                        f"""
                        ToolMessage class {tool_class} does not allow LLM use,
                        because `_allow_llm_use=False` either in the Tool or a 
                        parent class of this tool;
                        so not enabling LLM use for this tool!
                        If you intended an LLM to use this tool, 
                        set `_allow_llm_use=True` when you define the tool.
                        """
                    )
            else:
                self.llm_tools_usable.discard(t)
                self.llm_functions_usable.discard(t)

        # Set tool instructions and JSON format instructions
        if self.config.use_tools:
            self.system_tool_format_instructions = self.tool_format_rules()
        self.system_tool_instructions = self.tool_instructions()

    def disable_message_handling(
        self,
        message_class: Optional[Type[ToolMessage]] = None,
    ) -> None:
        """
        Disable this agent from RESPONDING to a `message_class` (Tool). If
            `message_class` is None, then disable this agent from responding to ALL.
        Args:
            message_class: The ToolMessage class to disable; Optional.
        """
        super().disable_message_handling(message_class)
        for t in self._get_tool_list(message_class):
            self.llm_tools_handled.discard(t)
            self.llm_functions_handled.discard(t)

    def disable_message_use(
        self,
        message_class: Optional[Type[ToolMessage]],
    ) -> None:
        """
        Disable this agent from USING a message class (Tool).
        If `message_class` is None, then disable this agent from USING ALL tools.
        Args:
            message_class: The ToolMessage class to disable.
                If None, disable all.
        """
        for t in self._get_tool_list(message_class):
            self.llm_tools_usable.discard(t)
            self.llm_functions_usable.discard(t)

    def disable_message_use_except(self, message_class: Type[ToolMessage]) -> None:
        """
        Disable this agent from USING ALL messages EXCEPT a message class (Tool)
        Args:
            message_class: The only ToolMessage class to allow
        """
        request = message_class.__fields__["request"].default
        to_remove = [r for r in self.llm_tools_usable if r != request]
        for r in to_remove:
            self.llm_tools_usable.discard(r)
            self.llm_functions_usable.discard(r)

    def truncate_message(
        self,
        idx: int,
        tokens: int = 5,
        warning: str = "...[Contents truncated!]",
    ) -> LLMMessage:
        """Truncate message at idx in msg history to `tokens` tokens"""
        llm_msg = self.message_history[idx]
        orig_content = llm_msg.content
        new_content = (
            self.parser.truncate_tokens(orig_content, tokens)
            if self.parser is not None
            else orig_content[: tokens * 4]  # approx truncation
        )
        llm_msg.content = new_content + "\n" + warning
        return llm_msg

    def _reduce_raw_tool_results(self, message: ChatDocument) -> None:
        """
        If message is the result of a ToolMessage that had
        a `_max_retained_tokens` set to a non-None value, then we replace contents
        with a placeholder message.
        """
        parent_message: ChatDocument | None = message.parent
        tools = [] if parent_message is None else parent_message.tool_messages
        truncate_tools = [t for t in tools if t._max_retained_tokens is not None]
        limiting_tool = truncate_tools[0] if len(truncate_tools) > 0 else None
        if limiting_tool is not None and limiting_tool._max_retained_tokens is not None:
            tool_name = limiting_tool.default_value("request")
            max_tokens: int = limiting_tool._max_retained_tokens
            truncation_warning = f"""
                The result of the {tool_name} tool were too large, 
                and has been truncated to {max_tokens} tokens.
                To obtain the full result, the tool needs to be re-used.
            """
            self.truncate_message(
                message.metadata.msg_idx, max_tokens, truncation_warning
            )

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Respond to a single user message, appended to the message history,
        in "chat" mode
        Args:
            message (str|ChatDocument): message or ChatDocument object to respond to.
                If None, use the self.task_messages
        Returns:
            LLM response as a ChatDocument object
        """
        if self.llm is None:
            return None
        hist, output_len = self._prep_llm_messages(message)
        if len(hist) == 0:
            return None
        tool_choice = (
            "auto"
            if isinstance(message, str)
            else (message.oai_tool_choice if message is not None else "auto")
        )
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):
            response = self.llm_response_messages(hist, output_len, tool_choice)
        self.message_history.extend(ChatDocument.to_LLMMessage(response))
        response.metadata.msg_idx = len(self.message_history) - 1
        response.metadata.agent_id = self.id
        if isinstance(message, ChatDocument):
            self._reduce_raw_tool_results(message)
        # Preserve trail of tool_ids for OpenAI Assistant fn-calls
        response.metadata.tool_ids = (
            []
            if isinstance(message, str)
            else message.metadata.tool_ids if message is not None else []
        )
        return response

    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Async version of `llm_response`. See there for details.
        """
        if self.llm is None:
            return None
        hist, output_len = self._prep_llm_messages(message)
        if len(hist) == 0:
            return None
        tool_choice = (
            "auto"
            if isinstance(message, str)
            else (message.oai_tool_choice if message is not None else "auto")
        )
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):
            response = await self.llm_response_messages_async(
                hist, output_len, tool_choice
            )
        self.message_history.extend(ChatDocument.to_LLMMessage(response))
        response.metadata.msg_idx = len(self.message_history) - 1
        response.metadata.agent_id = self.id
        if isinstance(message, ChatDocument):
            self._reduce_raw_tool_results(message)
        # Preserve trail of tool_ids for OpenAI Assistant fn-calls
        response.metadata.tool_ids = (
            []
            if isinstance(message, str)
            else message.metadata.tool_ids if message is not None else []
        )
        return response

    def init_message_history(self) -> None:
        """
        Initialize the message history with the system message and user message
        """
        self.message_history = [self._create_system_and_tools_message()]
        if self.user_message:
            self.message_history.append(
                LLMMessage(role=Role.USER, content=self.user_message)
            )

    def _prep_llm_messages(
        self,
        message: Optional[str | ChatDocument] = None,
        truncate: bool = True,
    ) -> Tuple[List[LLMMessage], int]:
        """
        Prepare messages to be sent to self.llm_response_messages,
            which is the main method that calls the LLM API to get a response.

        Returns:
            Tuple[List[LLMMessage], int]: (messages, output_len)
                messages = Full list of messages to send
                output_len = max expected number of tokens in response
        """

        if (
            not self.llm_can_respond(message)
            or self.config.llm is None
            or self.llm is None
        ):
            return [], 0

        if message is None and len(self.message_history) > 0:
            # this means agent has been used to get LLM response already,
            # and so the last message is an "assistant" response.
            # We delete this last assistant response and re-generate it.
            self.clear_history(-1)
            logger.warning(
                "Re-generating the last assistant response since message is None"
            )

        if len(self.message_history) == 0:
            # initial messages have not yet been loaded, so load them
            self.init_message_history()

            # for debugging, show the initial message history
            if settings.debug:
                print(
                    f"""
                [grey37]LLM Initial Msg History:
                {escape(self.message_history_str())}
                [/grey37]
                """
                )
        else:
            assert self.message_history[0].role == Role.SYSTEM
            # update the system message with the latest tool instructions
            self.message_history[0] = self._create_system_and_tools_message()

        if message is not None:
            if (
                isinstance(message, str)
                or message.id() != self.message_history[-1].chat_document_id
            ):
                # either the message is a str, or it is a fresh ChatDocument
                # different from the last message in the history
                llm_msgs = ChatDocument.to_LLMMessage(message, self.oai_tool_calls)
                # LLM only responds to the content, so only those msgs with
                # non-empty content should be kept
                llm_msgs = [m for m in llm_msgs if m.content.strip() != ""]
                if len(llm_msgs) == 0:
                    return [], 0
                # process tools if any
                done_tools = [m.tool_call_id for m in llm_msgs if m.role == Role.TOOL]
                self.oai_tool_calls = [
                    t for t in self.oai_tool_calls if t.id not in done_tools
                ]
                self.message_history.extend(llm_msgs)

        hist = self.message_history
        output_len = self.config.llm.max_output_tokens
        if (
            truncate
            and self.chat_num_tokens(hist)
            > self.llm.chat_context_length() - self.config.llm.max_output_tokens
        ):
            # chat + output > max context length,
            # so first try to shorten requested output len to fit.
            output_len = self.llm.chat_context_length() - self.chat_num_tokens(hist)
            if output_len < self.config.llm.min_output_tokens:
                # unacceptably small output len, so drop early parts of conv history
                # if output_len is still too long, then drop early parts of conv history
                # TODO we should really be doing summarization or other types of
                #   prompt-size reduction
                while (
                    self.chat_num_tokens(hist)
                    > self.llm.chat_context_length() - self.config.llm.min_output_tokens
                ):
                    # try dropping early parts of conv history
                    # TODO we should really be doing summarization or other types of
                    #   prompt-size reduction
                    if len(hist) <= 2:
                        # We want to preserve the first message (typically system msg)
                        # and last message (user msg).
                        raise ValueError(
                            """
                        The message history is longer than the max chat context 
                        length allowed, and we have run out of messages to drop.
                        HINT: In your `OpenAIGPTConfig` object, try increasing
                        `chat_context_length` or decreasing `max_output_tokens`.
                        """
                        )
                    # drop the second message, i.e. first msg after the sys msg
                    # (typically user msg).
                    ChatDocument.delete_id(hist[1].chat_document_id)
                    hist = hist[:1] + hist[2:]

                if len(hist) < len(self.message_history):
                    msg_tokens = self.chat_num_tokens()
                    logger.warning(
                        f"""
                    Chat Model context length is {self.llm.chat_context_length()} 
                    tokens, but the current message history is {msg_tokens} tokens long.
                    Dropped the {len(self.message_history) - len(hist)} messages
                    from early in the conversation history so that history token 
                    length is {self.chat_num_tokens(hist)}.
                    This may still not be low enough to allow minimum output length of 
                    {self.config.llm.min_output_tokens} tokens.
                    """
                    )

        if output_len < 0:
            raise ValueError(
                f"""
                Tried to shorten prompt history for chat mode 
                but even after dropping all messages except system msg and last (
                user) msg, the history token len {self.chat_num_tokens(hist)} is longer 
                than the model's max context length {self.llm.chat_context_length()}.
                Please try shortening the system msg or user prompts.
                """
            )
        if output_len < self.config.llm.min_output_tokens:
            logger.warning(
                f"""
                Tried to shorten prompt history for chat mode 
                but the feasible output length {output_len} is still
                less than the minimum output length {self.config.llm.min_output_tokens}.
                Your chat history is too long for this model, 
                and the response may be truncated.
                """
            )
        if isinstance(message, ChatDocument):
            # record the position of the corresponding LLMMessage in
            # the message_history
            message.metadata.msg_idx = len(hist) - 1
            message.metadata.agent_id = self.id

        return hist, output_len

    def _function_args(
        self,
    ) -> Tuple[
        Optional[List[LLMFunctionSpec]],
        str | Dict[str, str],
        Optional[List[OpenAIToolSpec]],
        Optional[Dict[str, Dict[str, str] | str]],
    ]:
        """Get function/tool spec arguments for OpenAI-compatible LLM API call"""
        functions: Optional[List[LLMFunctionSpec]] = None
        fun_call: str | Dict[str, str] = "none"
        tools: Optional[List[OpenAIToolSpec]] = None
        force_tool: Optional[Dict[str, Dict[str, str] | str]] = None
        if self.config.use_functions_api and len(self.llm_functions_usable) > 0:
            if not self.config.use_tools_api:
                functions = [
                    self.llm_functions_map[f] for f in self.llm_functions_usable
                ]
                fun_call = (
                    "auto"
                    if self.llm_function_force is None
                    else self.llm_function_force
                )
            else:
                tools = [
                    OpenAIToolSpec(type="function", function=self.llm_functions_map[f])
                    for f in self.llm_functions_usable
                ]
                force_tool = (
                    None
                    if self.llm_function_force is None
                    else {
                        "type": "function",
                        "function": {"name": self.llm_function_force["name"]},
                    }
                )
        return functions, fun_call, tools, force_tool

    def llm_response_messages(
        self,
        messages: List[LLMMessage],
        output_len: Optional[int] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
    ) -> ChatDocument:
        """
        Respond to a series of messages, e.g. with OpenAI ChatCompletion
        Args:
            messages: seq of messages (with role, content fields) sent to LLM
            output_len: max number of tokens expected in response.
                    If None, use the LLM's default max_output_tokens.
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        assert self.config.llm is not None and self.llm is not None
        output_len = output_len or self.config.llm.max_output_tokens
        streamer = noop_fn
        if self.llm.get_stream():
            streamer = self.callbacks.start_llm_stream()
        self.llm.config.streamer = streamer
        with ExitStack() as stack:  # for conditionally using rich spinner
            if not self.llm.get_stream() and not settings.quiet:
                # show rich spinner only if not streaming!
                # (Why? b/c the intent of showing a spinner is to "show progress",
                # and we don't need to do that when streaming, since
                # streaming output already shows progress.)
                cm = status(
                    "LLM responding to messages...",
                    log_if_quiet=False,
                )
                stack.enter_context(cm)
            if self.llm.get_stream() and not settings.quiet:
                console.print(f"[green]{self.indent}", end="")
            functions, fun_call, tools, force_tool = self._function_args()
            assert self.llm is not None
            response = self.llm.chat(
                messages,
                output_len,
                tools=tools,
                tool_choice=force_tool or tool_choice,
                functions=functions,
                function_call=fun_call,
            )
        if self.llm.get_stream():
            self.callbacks.finish_llm_stream(
                content=str(response),
                is_tool=self.has_tool_message_attempt(
                    ChatDocument.from_LLMResponse(response, displayed=True),
                ),
            )
        self.llm.config.streamer = noop_fn
        if response.cached:
            self.callbacks.cancel_llm_stream()
        self._render_llm_response(response)
        self.update_token_usage(
            response,  # .usage attrib is updated!
            messages,
            self.llm.get_stream(),
            chat=True,
            print_response_stats=self.config.show_stats and not settings.quiet,
        )
        chat_doc = ChatDocument.from_LLMResponse(response, displayed=True)
        self.oai_tool_calls = response.oai_tool_calls or []
        self.oai_tool_id2call.update(
            {t.id: t for t in self.oai_tool_calls if t.id is not None}
        )
        return chat_doc

    async def llm_response_messages_async(
        self,
        messages: List[LLMMessage],
        output_len: Optional[int] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
    ) -> ChatDocument:
        """
        Async version of `llm_response_messages`. See there for details.
        """
        assert self.config.llm is not None and self.llm is not None
        output_len = output_len or self.config.llm.max_output_tokens
        functions, fun_call, tools, force_tool = self._function_args()
        assert self.llm is not None

        streamer_async = async_noop_fn
        if self.llm.get_stream():
            streamer_async = await self.callbacks.start_llm_stream_async()
        self.llm.config.streamer_async = streamer_async

        response = await self.llm.achat(
            messages,
            output_len,
            tools=tools,
            tool_choice=force_tool or tool_choice,
            functions=functions,
            function_call=fun_call,
        )
        if self.llm.get_stream():
            self.callbacks.finish_llm_stream(
                content=str(response),
                is_tool=self.has_tool_message_attempt(
                    ChatDocument.from_LLMResponse(response, displayed=True),
                ),
            )
        self.llm.config.streamer_async = async_noop_fn
        if response.cached:
            self.callbacks.cancel_llm_stream()
        self._render_llm_response(response)
        self.update_token_usage(
            response,  # .usage attrib is updated!
            messages,
            self.llm.get_stream(),
            chat=True,
            print_response_stats=self.config.show_stats and not settings.quiet,
        )
        chat_doc = ChatDocument.from_LLMResponse(response, displayed=True)
        self.oai_tool_calls = response.oai_tool_calls or []
        self.oai_tool_id2call.update(
            {t.id: t for t in self.oai_tool_calls if t.id is not None}
        )
        return chat_doc

    def _render_llm_response(
        self, response: ChatDocument | LLMResponse, citation_only: bool = False
    ) -> None:
        is_cached = (
            response.cached
            if isinstance(response, LLMResponse)
            else response.metadata.cached
        )
        if self.llm is None:
            return
        if not citation_only and (not self.llm.get_stream() or is_cached):
            # We would have already displayed the msg "live" ONLY if
            # streaming was enabled, AND we did not find a cached response.
            # If we are here, it means the response has not yet been displayed.
            cached = f"[red]{self.indent}(cached)[/red]" if is_cached else ""
            if not settings.quiet:
                chat_doc = (
                    response
                    if isinstance(response, ChatDocument)
                    else ChatDocument.from_LLMResponse(response, displayed=True)
                )
                # TODO: prepend TOOL: or OAI-TOOL: if it's a tool-call
                print(cached + "[green]" + escape(str(response)))
                self.callbacks.show_llm_response(
                    content=str(response),
                    is_tool=self.has_tool_message_attempt(chat_doc),
                    cached=is_cached,
                )
        if isinstance(response, LLMResponse):
            # we are in the context immediately after an LLM responded,
            # we won't have citations yet, so we're done
            return
        if response.metadata.has_citation and not settings.quiet:
            print("[grey37]SOURCES:\n" + escape(response.metadata.source) + "[/grey37]")
            self.callbacks.show_llm_response(
                content=str(response.metadata.source),
                is_tool=False,
                cached=False,
                language="text",
            )

    def _llm_response_temp_context(self, message: str, prompt: str) -> ChatDocument:
        """
        Get LLM response to `prompt` (which presumably includes the `message`
        somewhere, along with possible large "context" passages),
        but only include `message` as the USER message, and not the
        full `prompt`, in the message history.
        Args:
            message: the original, relatively short, user request or query
            prompt: the full prompt potentially containing `message` plus context

        Returns:
            Document object containing the response.
        """
        # we explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):  # type: ignore
            answer_doc = cast(ChatDocument, ChatAgent.llm_response(self, prompt))
        self.update_last_message(message, role=Role.USER)
        return answer_doc

    async def _llm_response_temp_context_async(
        self, message: str, prompt: str
    ) -> ChatDocument:
        """
        Async version of `_llm_response_temp_context`. See there for details.
        """
        # we explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):  # type: ignore
            answer_doc = cast(
                ChatDocument,
                await ChatAgent.llm_response_async(self, prompt),
            )
        self.update_last_message(message, role=Role.USER)
        return answer_doc

    def llm_response_forget(self, message: str) -> ChatDocument:
        """
        LLM Response to single message, and restore message_history.
        In effect a "one-off" message & response that leaves agent
        message history state intact.

        Args:
            message (str): user message

        Returns:
            A Document object with the response.

        """
        # explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        n_msgs = len(self.message_history)
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):  # type: ignore
            response = cast(ChatDocument, ChatAgent.llm_response(self, message))
        # If there is a response, then we will have two additional
        # messages in the message history, i.e. the user message and the
        # assistant response. We want to (carefully) remove these two messages.
        if len(self.message_history) > n_msgs:
            msg = self.message_history.pop()
            self._drop_msg_update_tool_calls(msg)

        if len(self.message_history) > n_msgs:
            msg = self.message_history.pop()
            self._drop_msg_update_tool_calls(msg)

        return response

    async def llm_response_forget_async(self, message: str) -> ChatDocument:
        """
        Async version of `llm_response_forget`. See there for details.
        """
        # explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        n_msgs = len(self.message_history)
        with StreamingIfAllowed(self.llm, self.llm.get_stream()):  # type: ignore
            response = cast(
                ChatDocument, await ChatAgent.llm_response_async(self, message)
            )
        # If there is a response, then we will have two additional
        # messages in the message history, i.e. the user message and the
        # assistant response. We want to (carefully) remove these two messages.
        if len(self.message_history) > n_msgs:
            msg = self.message_history.pop()
            self._drop_msg_update_tool_calls(msg)

        if len(self.message_history) > n_msgs:
            msg = self.message_history.pop()
            self._drop_msg_update_tool_calls(msg)
        return response

    def chat_num_tokens(self, messages: Optional[List[LLMMessage]] = None) -> int:
        """
        Total number of tokens in the message history so far.

        Args:
            messages: if provided, compute the number of tokens in this list of
                messages, rather than the current message history.
        Returns:
            int: number of tokens in message history
        """
        if self.parser is None:
            raise ValueError(
                "ChatAgent.parser is None. "
                "You must set ChatAgent.parser "
                "before calling chat_num_tokens()."
            )
        hist = messages if messages is not None else self.message_history
        return sum([self.parser.num_tokens(m.content) for m in hist])

    def message_history_str(self, i: Optional[int] = None) -> str:
        """
        Return a string representation of the message history
        Args:
            i: if provided, return only the i-th message when i is postive,
                or last k messages when i = -k.
        Returns:
        """
        if i is None:
            return "\n".join([str(m) for m in self.message_history])
        elif i > 0:
            return str(self.message_history[i])
        else:
            return "\n".join([str(m) for m in self.message_history[i:]])
