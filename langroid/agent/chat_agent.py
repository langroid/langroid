import logging
from contextlib import ExitStack
from typing import Dict, List, Optional, Set, Type, cast, no_type_check

from rich import print
from rich.console import Console

from langroid.agent.base import Agent, AgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.base import (
    LanguageModel,
    LLMFunctionSpec,
    LLMMessage,
    Role,
    StreamingIfAllowed,
)
from langroid.utils.configuration import settings

console = Console()

logger = logging.getLogger(__name__)


class ChatAgentConfig(AgentConfig):
    """
    Configuration for ChatAgent
    Attributes:
        system_message: system message to include in message sequence
             (typically defines role and task of agent)
        user_message: user message to include in message sequence
        use_tools: whether to use our own ToolMessages mechanism
        use_functions_api: whether to use functions native to the LLM API
                (e.g. OpenAI's `function_call` mechanism)
    """

    system_message: str = "You are a helpful assistant."
    user_message: Optional[str] = None
    use_tools: bool = True
    use_functions_api: bool = False


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
        self, config: ChatAgentConfig, task: Optional[List[LLMMessage]] = None
    ):
        """
        Chat-mode agent initialized with task spec as the initial message sequence
        Args:
            config: settings for the agent

        """
        super().__init__(config)
        self.config: ChatAgentConfig = config
        self.message_history: List[LLMMessage] = []
        self.json_instructions_idx: int = -1
        self.llm_functions_map: Dict[str, LLMFunctionSpec] = {}
        self.llm_functions_handled: Set[str] = set()
        self.llm_functions_usable: Set[str] = set()
        self.llm_function_force: Optional[Dict[str, str]] = None

        priming_messages = task
        if priming_messages is None:
            priming_messages = [
                LLMMessage(role=Role.SYSTEM, content=config.system_message),
            ]
            if config.user_message:
                priming_messages.append(
                    LLMMessage(role=Role.USER, content=config.user_message)
                )
        self.task_messages = priming_messages

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

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the message history.
        Args:
            message (str): user message
        """
        if len(self.message_history) > 0:
            self.message_history.append(LLMMessage(role=Role.USER, content=message))
        else:
            self.task_messages.append(LLMMessage(role=Role.USER, content=message))

    def update_last_message(self, message: str, role: str = Role.USER) -> None:
        """
        Update the last message with role `role` in the message history.
        Useful when we want to replace a long user prompt, that may contain context
        documents plus a question, with just the question.
        Args:
            message (str): user message
            role (str): role of message to replace
        """
        if len(self.message_history) == 0:
            return
        # find last message in self.message_history with role `role`
        for i in range(len(self.message_history) - 1, -1, -1):
            if self.message_history[i].role == role:
                self.message_history[i].content = message
                break

    def enable_message(
        self,
        message_class: Optional[Type[ToolMessage]],
        use: bool = True,
        handle: bool = True,
        force: bool = False,
    ) -> None:
        """
        Add the tool (message class) to the agent, and enable either
        - tool USE (i.e. the LLM can generate JSON to use this tool),
        - tool HANDLING (i.e. the agent can handle JSON from this tool),

        Args:
            message_class: The ToolMessage class to enable,
                for USE, or HANDLING, or both.
                Optional; if None, then apply the enabling to all tools in the
                agent's toolset that have been enabled so far.
            use: IF True, allow the agent (LLM) to use this tool (or all tools),
                else disallow
            handle: if True, allow the agent (LLM) to handle (i.e. respond to) this
                tool (or all tools)
            force: whether to FORCE the agent (LLM) to USE the specific
                 tool represented by `message_class`.
                 `force` is ignored if `message_class` is None.

        """
        super().enable_message_handling(message_class)  # enables handling only
        tools = self._get_tool_list(message_class)
        if message_class is not None:
            request = message_class.default_value("request")
            llm_function = message_class.llm_function_schema()
            self.llm_functions_map[request] = llm_function
            if force:
                self.llm_function_force = dict(name=llm_function.name)
            else:
                self.llm_function_force = None
        n_usable_tools = len(self.llm_tools_usable)
        for t in tools:
            if handle:
                self.llm_tools_handled.add(t)
                self.llm_functions_handled.add(t)
            else:
                self.llm_tools_handled.discard(t)
                self.llm_functions_handled.discard(t)

            if use:
                self.llm_tools_usable.add(t)
                self.llm_functions_usable.add(t)
            else:
                self.llm_tools_usable.discard(t)
                self.llm_functions_usable.discard(t)

        # TODO we should do this only on demand when we actually are
        # ready to send the instructions.
        # But for now leave as is.
        if len(self.llm_tools_usable) != n_usable_tools and self.config.use_tools:
            # Update JSON format instructions if the set of usable tools has changed
            self.update_message_instructions()

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
        for r in self.llm_functions_usable:
            if r != request:
                self.llm_tools_usable.discard(r)
                self.llm_functions_usable.discard(r)

    def update_message_instructions(self) -> None:
        """
        Add special instructions on situations when the LLM should send JSON-formatted
        messages, and save the index position of these instructions in the
        message history.
        """
        # note according to the openai-cookbook, GPT-3.5 pays less attention to the
        # system messages, so we add the instructions as a user message
        # TODO need to adapt this based on model type.
        json_instructions = super().message_format_instructions()
        if self.json_instructions_idx < 0:
            self.task_messages.append(
                LLMMessage(role=Role.USER, content=json_instructions)
            )
            self.json_instructions_idx = len(self.task_messages) - 1
        else:
            self.task_messages[self.json_instructions_idx].content = json_instructions

        # Note that task_messages is the initial set of messages created to set up
        # the task, and they may not yet have been sent to the LLM at this point.

        # But if the task_messages have already been sent to the LLM, then we need to
        # update the self.message_history as well, since this history will be sent to
        # the LLM on each round, after appending the latest assistant, user msgs.
        if len(self.message_history) > 0:
            self.message_history[self.json_instructions_idx].content = json_instructions

    @no_type_check
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
        if not self.llm_can_respond(message):
            return None

        assert (
            message is not None or len(self.message_history) == 0
        ), "message can be None only if message_history is empty, i.e. at start."

        if len(self.message_history) == 0:
            # task_messages have not yet been loaded, so load them
            self.message_history = self.task_messages.copy()
            # for debugging, show the initial message history
            if settings.debug:
                print(
                    f"""
                [red]LLM Initial Msg History:
                {self.message_history_str()}
                """
                )

        if message is not None:
            llm_msg = ChatDocument.to_LLMMessage(message)
            self.message_history.append(llm_msg)

        hist = self.message_history
        output_len = self.config.llm.max_output_tokens
        if (
            self.chat_num_tokens(hist)
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
                        # first two are "reserved" for the initial system, user msgs
                        # that presumably set up the initial "priming" or "task" for
                        # the agent.
                        raise ValueError(
                            """
                        The message history is longer than the max chat context 
                        length allowed, and we have run out of messages to drop."""
                        )
                    hist = hist[:2] + hist[3:]

                if len(hist) < len(self.message_history):
                    msg_tokens = self.chat_num_tokens()
                    logger.warning(
                        f"""
                    Chat Model context length is {self.llm.chat_context_length()} 
                    tokens, but the current message history is {msg_tokens} tokens long.
                    Dropped the {len(self.message_history) - len(hist)} messages
                    from early in the conversation history so total tokens are 
                    low enough to allow minimum output length of 
                    {self.config.llm.min_output_tokens} tokens.
                    """
                    )

        if output_len < self.config.llm.min_output_tokens:
            raise ValueError(
                f"""
                Tried to shorten prompt history for chat mode 
                but the feasible output length {output_len} is still
                less than the minimum output length {self.config.llm.min_output_tokens}.
                """
            )
        with StreamingIfAllowed(self.llm):
            response = self.llm_response_messages(hist, output_len)
        # TODO - when response contains function_call we should include
        # that (and related fields) in the message_history
        self.message_history.append(ChatDocument.to_LLMMessage(response))
        return response

    def llm_response_messages(
        self, messages: List[LLMMessage], output_len: Optional[int] = None
    ) -> ChatDocument:
        """
        Respond to a series of messages, e.g. with OpenAI ChatCompletion
        Args:
            messages: seq of messages (with role, content fields) sent to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        assert self.config.llm is not None and self.llm is not None
        output_len = output_len or self.config.llm.max_output_tokens
        with ExitStack() as stack:  # for conditionally using rich spinner
            if not self.llm.get_stream():  # type: ignore
                # show rich spinner only if not streaming!
                cm = console.status("LLM responding to messages...")
                stack.enter_context(cm)
            if self.llm.get_stream():  # type: ignore
                console.print(f"[green]{self.indent}", end="")
            functions: Optional[List[LLMFunctionSpec]] = None
            fun_call: str | Dict[str, str] = "none"
            if self.config.use_functions_api and len(self.llm_functions_usable) > 0:
                functions = [
                    self.llm_functions_map[f] for f in self.llm_functions_usable
                ]
                fun_call = (
                    "auto"
                    if self.llm_function_force is None
                    else self.llm_function_force
                )
            assert self.llm is not None
            response = cast(LanguageModel, self.llm).chat(
                messages,
                output_len,
                functions=functions,
                function_call=fun_call,
            )
        displayed = False
        if not self.llm.get_stream() or response.cached:  # type: ignore
            displayed = True
            cached = f"[red]{self.indent}(cached)[/red]" if response.cached else ""
            if response.function_call is not None:
                response_str = str(response.function_call)
            else:
                response_str = response.message
            print(cached + "[green]" + response_str)

        return ChatDocument.from_LLMResponse(response, displayed)

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
        answer_doc = cast(ChatDocument, ChatAgent.llm_response(self, prompt))
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
        response = cast(ChatDocument, ChatAgent.llm_response(self, message))
        # clear the last two messages, which are the
        # user message and the assistant response
        self.message_history.pop()
        self.message_history.pop()
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
