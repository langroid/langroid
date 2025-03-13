import logging
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, no_type_check

from anthropic import (
    Anthropic,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
    UnprocessableEntityError,
)
from anthropic.types import (
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from rich import print
from rich.markup import escape

from langroid.language_models import (
    LLMConfig,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    StreamEventType,
)
from langroid.language_models.base import (
    AnthropicSystemCacheControl,
    AnthropicToolSpec,
    LanguageModel,
    LLMCallConfig,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
    PromptVariants,
    Role,
    ToolChoiceTypes,
    ToolVariantSelector,
    filter_default_model,
)
from langroid.language_models.model_info import (
    AnthropicModel as AnthropicChatModels,
)
from langroid.language_models.openai_gpt import available_models
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    base_retry_errors,
    retry_with_exponential_backoff,
)
from langroid.pydantic_v1 import BaseModel
from langroid.utils.configuration import settings
from langroid.utils.constants import Colors

logging.getLogger("anthropic").setLevel(logging.ERROR)


ANTHROPIC_API_KEY = ""
DUMMY_API_KEY = ""

anthropic_chat_model_pref_list = [
    AnthropicChatModels.CLAUDE_3_5_HAIKU,
    AnthropicChatModels.CLAUDE_3_5_SONNET,
    AnthropicChatModels.CLAUDE_3_HAIKU,
]

default_anthropic_chat_model = filter_default_model(
    anthropic_chat_model_pref_list,
    available_models,
    [AnthropicChatModels.CLAUDE_3_5_HAIKU],
)

retryable_anthropic_errors = base_retry_errors + (
    APITimeoutError,
    RateLimitError,
    APIError,
)

terminal_anthropic_errors = (
    BadRequestError,
    AuthenticationError,
    UnprocessableEntityError,
)


class AnthropicCitationBase(BaseModel):
    cited_text: str
    document_index: int
    document_title: str


class AnthropicCitationRequestCharLocation(AnthropicCitationBase):
    end_char_index: int
    start_char_index: int
    type: str = "char_location"


class AnthropicRequestPageLocation(AnthropicCitationBase):
    end_page_number: int
    start_page_number: int
    type: str = "page_location"


class AnthropicRequestContentBlockLocation(AnthropicCitationBase):
    end_block_index: int
    start_block_index: int
    type: str = "content_block_location"


class AnthropicSystemMessage(BaseModel):
    type: str = "text"
    text: str = "You are a helpful assistant."
    cache_control: Optional[AnthropicSystemCacheControl] = None
    citation: Optional[AnthropicCitationBase] = None


class AnthropicSystemConfig(BaseModel):
    system_prompts: str | List[AnthropicSystemMessage] = "You are a helpful assistant."

    def prepare_system_prompt(self) -> Union[str, list[Dict[str, Any]]]:
        if isinstance(self.system_prompts, list):
            return [prompt.dict() for prompt in self.system_prompts]
        return self.system_prompts


class AnthropicToolChoice(BaseModel):
    """
    Class defining the tool choice configuration
    for an Anthropic Messages API call.
    The values provided are the defaults set by
    Anthropic: https://docs.anthropic.com/en/api/messages#body-tool-choice
    """

    type: str = "auto"
    disable_parallel_tool_use: bool = False


class AnthropicToolCall(BaseModel):
    id: str
    name: str


class AnthropicStopReason(Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class StreamTypes(Enum):
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_STOP = "content_block_stop"
    ERROR = "error"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    MESSAGE_START = "message_start"
    PING = "ping"


class Delta(Enum):
    TEXT = "text_delta"
    JSON = "input_json_delta"
    THINK = "thinking_delta"
    SIGNATURE = "signature_delta"


class AnthropicStreamDeltaEvent(BaseModel):
    delta: Optional[Literal[StreamTypes.CONTENT_BLOCK_DELTA]]
    thinking: str
    text: str


class StreamState(BaseModel):
    """
    Class defining unpacked values from Anthropic SSE
    for internal stream processing.
    """

    terminal_state: bool


class AnthropicCallParams(LLMCallConfig):
    """
    Parameters describing defaults used when performing Anthropic chat-completion calls.
    When specified, any param here overrides the one with the same name in the
    Anthropic config.
    See Anthropic's API reference documentation for details on available params:
    https://docs.anthropic.com/en/api/messages
    """


class AnthropicLLMConfig(LLMConfig):
    """
    Class for Anthropic API configuration
    """

    type: str = "anthropic"
    api_key: str = DUMMY_API_KEY
    organization: str = ""
    litellm: bool = False
    ollama = False
    min_output_tokens = 1
    temperature = 0.2
    seed: int | None = 42
    params: AnthropicCallParams | None = None
    chat_model: str = default_anthropic_chat_model
    system_config: AnthropicSystemConfig | None = None

    def __init__(self, **kwargs) -> None:  # type: ignore
        super().__init__(**kwargs)


class AnthropicLLM(LanguageModel):
    """
    Class for Anthropic LLMs
    """

    client: Anthropic | None
    async_client: AsyncAnthropic | None

    def __init__(self, config: AnthropicLLMConfig = AnthropicLLMConfig()):
        config = config.copy()
        super().__init__(config)
        self.config: AnthropicLLMConfig = config
        self.chat_model_orig = self.config.chat_model

        # check if creation hook function needed

        if settings.chat_model != "":
            self.config.chat_model = settings.chat_model
            self.chat_model_orig = settings.chat_model
            self.config.completion_model = settings.chat_model

        # alternate formatter or HF formatter check?

        if settings.chat_model != "":
            self.config.completion_model = self.config.chat_model

        if self.config.formatter is not None:
            self.config.use_completion_for_chat = True
            self.config.completion_model = self.config.chat_model

        if self.config.use_completion_for_chat:
            self.config.use_chat_for_completion = False

        self.api_key = config.api_key
        if self.api_key == DUMMY_API_KEY:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", DUMMY_API_KEY)

        self.client = Anthropic(
            api_key=self.api_key, timeout=Timeout(timeout=self.config.timeout)
        )
        self.async_client = AsyncAnthropic(
            api_key=self.api_key, timeout=Timeout(timeout=self.config.timeout)
        )

    def generate(
        self,
        prompt: str,
        prompt_variants: PromptVariants = PromptVariants(),
        max_tokens: int = 200,
    ) -> LLMResponse:
        """
        Entry function for chat completions. Anthropic used to have an API
        endpoint for 'Text Completions':
        https://docs.anthropic.com/en/api/complete
        However, this has been marked as legacy. They now direct
        completions via their Messages API in their documentation.
        The `generate` code path leverages that methodology.
        """
        try:
            return self._generate(prompt_variants, max_tokens)
        except Exception as e:
            logging.error("Error in AnthropicLLM::generate: ")
            logging.error(e)
            raise e

    def _generate(
        self, prompt_variants: PromptVariants, max_tokens: int
    ) -> LLMResponse:
        if not prompt_variants.anthropic:
            raise ValueError("Empty prompt list passed into generate")

        messages = [
            LLMMessage.parse_obj(prompt) for prompt in prompt_variants.anthropic
        ]

        if settings.debug:
            # using Claude 3.x+ models, the completion prompt should always be the last
            # message with a type of "assistant"
            message: LLMMessage = messages[-1]
            print(f"[grey37]ROLE: {message.role}[/grey37]")
            print(f"[grey37]PROMPT: {escape(message.content)}[/grey37]")

        call_dict = dict(
            model=self.config.completion_model,
            messages=messages,
            max_tokens=max_tokens,
            stream=self.get_stream(),
        )

        # if user has configured self.config.params, we should
        # now place those values into the completion call
        hydrated_call_dict = self._hydrate_completion_call(completion_call=call_dict)

        result: LLMResponse = self._completions_with_backoff(**hydrated_call_dict)
        return result

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        return LLMResponse(message="TODO")

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "",
        response_format: OpenAIJsonSchemaSpec | None = None,
        tool_variants: ToolVariantSelector = ToolVariantSelector(
            open_ai=[], anthropic=[]
        ),
    ) -> LLMResponse:
        """
        Get chat-completion response from an Anthropic API call

        Args:
            messages: message history for call
            max_tokens: max allotment of tokens for call
            tools: available registered tools for usage in response
            tool_choice: "auto", "any", or "tool"
            functions: not available with Anthropic's message API
            function_call: not available with Anthropic's message API
            response_format: not available with Anthropic's message API
            tool_variants: list of dicts that can be parsed into tool specs
                for Anthropic
        """

        if functions or function_call:
            logging.warning(
                """You might observe an unexpected answer with function usage
                as they are unavailable for Anthropic SDK calls."""
            )
            logging.warning("Please use the tools and tool_choice parameters instead.")

        try:
            return self._chat_helper(
                messages=messages,
                max_tokens=max_tokens,
                tools=tool_variants.anthropic,
                tool_choice=tool_choice,
            )
        except Exception as e:
            logging.error("Error in AnthropicLLM::chat")
            raise e

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "",
        response_format: OpenAIJsonSchemaSpec | None = None,
        tool_variants: ToolVariantSelector = ToolVariantSelector(
            open_ai=[], anthropic=[]
        ),
    ) -> LLMResponse:
        if functions or function_call:
            logging.warning(
                """You might observe an unexpected answer with function usage
                as they are unavailable for Anthropic SDK calls."""
            )
            logging.warning("Please use the tools and tool_choice parameters instead.")

        try:
            result = await self._achat_helper(
                messages=messages,
                max_tokens=max_tokens,
                tools=tool_variants.anthropic,
                tool_choice=tool_choice,
            )
            return result
        except Exception as e:
            logging.error("Error in AnthropicLLM::achat")
            raise e

    async def _achat_helper(
        self,
        messages: str | List[LLMMessage],
        max_tokens: int,
        tools: Optional[List[AnthropicToolSpec]],
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
    ) -> LLMResponse:
        """
        Asynchronous chat completion API calls to Anthropic
        """
        args = self._prep_chat_args(
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = await self._achat_completion_with_retry_backoff(**args)
        if self.get_stream():
            anthropic_response: LLMResponse = await self._stream_response_async(
                response, chat=True
            )
            return anthropic_response
        if isinstance(response, dict):
            response_dict = response
        else:
            response_dict = response.model_dump()

        return self._process_chat_completion_response(response_dict=response_dict)

    @retry_with_exponential_backoff(
        retryable_errors=retryable_anthropic_errors,
        terminal_errors=terminal_anthropic_errors,
    )
    def _chat_completion_with_retry_backoff(self, **kwargs):  # type: ignore
        try:
            assert self.client
        except AssertionError as e:
            logging.error("Anthropic client is not defined on message call.")
            raise e
        messages_call = self.client.messages.create
        return messages_call(**kwargs)

    def _chat_helper(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        tools: Optional[List[AnthropicToolSpec]],
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]],
    ) -> LLMResponse:
        """
        Chat completion API calls to Anthropic
        """
        args = self._prep_chat_args(
            messages, max_tokens, tools=tools, tool_choice=tool_choice
        )
        response = self._chat_completion_with_retry_backoff(**args)
        # we need to check if it's a stream response
        if self.get_stream():
            stream_response: LLMResponse = self._stream_response(
                response=response, chat=True
            )
            return stream_response
        # else we parse the response and return it
        if isinstance(response, dict):
            response_dict = response
        else:
            response_dict = response.model_dump()
        return self._process_chat_completion_response(response_dict)

    @async_retry_with_exponential_backoff(
        retryable_errors=retryable_anthropic_errors,
        terminal_errors=terminal_anthropic_errors,
    )
    async def _achat_completion_with_retry_backoff(self, **kwargs):  # type: ignore
        try:
            assert self.async_client
        except AssertionError as e:
            logging.error("Anthropic async client is not defined on message call.")
            raise e
        async_completion_call = self.async_client.messages.create
        result = await async_completion_call(**kwargs)
        return result

    def _process_chat_completion_response(
        self, response_dict: Dict[str, Any]
    ) -> LLMResponse:
        """
        Example Anthropic Response for Messages API:
        {
            "content": [
                {
                    "text": "Hi! My name is Claude.",
                    "type": "text"
                }
            ],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-3-7-sonnet-20250219",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "type": "message",
            "usage": {
                "input_tokens": 2095,
                "output_tokens": 503
            }
        }
        """
        message = self._parse_content_from_response(response_dict=response_dict)
        usage = self._parse_usage_from_response(response_dict=response_dict)
        ant_tool_calls = None
        if response_dict.get("stop_reason", "") == AnthropicStopReason.TOOL_USE.value:
            ant_tool_calls = self._parse_tool_usage_from_response(
                response_dict=response_dict
            )
        return LLMResponse(
            message=message, anthropic_tool_calls=ant_tool_calls or None, usage=usage
        )

    @staticmethod
    def _parse_content_from_response(response_dict: Dict[str, Any]) -> str:
        """
        Parse the dict from a content item on an Anthropic Messages API
        call response.
        """
        content = response_dict.get("content", [])
        message = content[0].get("text") if content else ""
        return message

    @staticmethod
    def _parse_tool_usage_from_response(
        response_dict: Dict[str, Any]
    ) -> List[AnthropicToolCall]:
        content = response_dict.get("content", [])
        tools = []
        for block in content:
            if block.get("type") == AnthropicStopReason.TOOL_USE.value:
                tools.append(AnthropicToolCall.parse_obj(block))
        return tools

    def _parse_usage_from_response(
        self, response_dict: Dict[str, Any]
    ) -> LLMTokenUsage:
        cost = 0.0
        input_tokens = 0  # prompt_tokens
        output_tokens = 0  # completion_tokens
        if not self.get_stream() and response_dict.get("usage"):
            input_tokens = response_dict["usage"]["input_tokens"]
            output_tokens = response_dict["usage"]["output_tokens"]
            cost = self._calculate_cost(input_tokens, output_tokens)

        return LLMTokenUsage(
            prompt_tokens=input_tokens, completion_tokens=output_tokens, cost=cost
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        prompt, generation = self.chat_cost()
        return (prompt * input_tokens + generation * output_tokens) / 1000

    def _prep_chat_args(
        self,
        messages: str | List[LLMMessage],
        max_tokens: int,
        tools: Optional[List[AnthropicToolSpec]],
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Helper function to prepare payload to send to Anthropic's chat model API.
        """
        if isinstance(messages, str):
            # Anthropic does not utilize "system" roles
            llm_messages = [LLMMessage(role=Role.USER, content=messages)]
        else:
            llm_messages = messages

        chat_model = self.config.chat_model

        args: Dict[str, Any] = dict(
            model=chat_model,
            messages=[m.api_dict(has_system_role=False) for m in llm_messages],
            max_tokens=max_tokens,
            stream=self.get_stream(),
        )

        # Instead of a system role, Anthropic has an option to define
        # the system behavior separately in the API call; if a user
        # has set up the system config option, we populate it here
        if self.config.system_config and self.config.system_config.system_prompts:
            args["system"] = self.config.system_config.prepare_system_prompt()

        args["tool_choice"] = tool_choice

        if tools is not None:
            args["tools"] = [t.dict() for t in tools]

        # Anthropic does not have a response_format object
        # in their messages function call
        return args

    @retry_with_exponential_backoff(
        retryable_errors=retryable_anthropic_errors,
        terminal_errors=terminal_anthropic_errors,
    )
    def _stream_response(self, response, chat: bool = False) -> LLMResponse:  # type: ignore
        """
        Grab and print streaming response from Anthropic Messages API
        with server sent events (https://docs.anthropic.com/en/api/messages-streaming).
        If we get a call here, the response from the SDK call should be of type
        RawMessageStreamEvent.
        """
        completion = ""
        reasoning = ""
        tool_usage: Dict[int, AnthropicToolCall] = {}
        tool_partials: Dict[int, List[str]] = {}

        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()

        try:
            for event in response:
                synchronous_stream = self._process_stream_event(  # type: ignore
                    event=event,
                    tool_usage=tool_usage,
                    tool_partials=tool_partials,
                    chat=chat,
                    completion=completion,
                    reasoning=reasoning,
                )
                if synchronous_stream.terminal_state:
                    break
        except Exception:
            pass

        print("")

        return self._create_stream_response(
            tool_usage=tool_usage, completion=completion, reasoning=reasoning
        )

    @async_retry_with_exponential_backoff(
        retryable_errors=retryable_anthropic_errors,
        terminal_errors=terminal_anthropic_errors,
    )
    async def _stream_response_async(self, response, chat: bool = False) -> LLMResponse:  # type: ignore
        completion = ""
        reasoning = ""
        tool_usage: Dict[int, AnthropicToolCall] = {}
        tool_partials: Dict[int, List[str]] = {}

        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()

        try:
            async for event in response:
                asynchronous_stream = await self._process_stream_event_async(  # type: ignore
                    event=event,
                    tool_usage=tool_usage,
                    tool_partials=tool_partials,
                    chat=chat,
                    completion=completion,
                    reasoning=reasoning,
                )
                if asynchronous_stream.terminal_state:
                    break
        except Exception:
            pass

        print("")

        return self._create_stream_response(
            tool_usage=tool_usage, completion=completion, reasoning=reasoning
        )

    @staticmethod
    def _handle_early_stream_terminations(event, tool_usage) -> Optional[StreamState]:  # type: ignore
        if event.type in [
            StreamTypes.PING.value,
            StreamTypes.MESSAGE_START.value,
        ]:
            return StreamState(
                terminal_state=False,
            )

        # we need a different check for content_block_start
        # because Anthropic can have a tool_use call here,
        # and this is the response they place metadata for tool
        # usage such as tool_id and tool_name
        if event.type == StreamTypes.CONTENT_BLOCK_START.value:
            if isinstance(event.content_block, ToolUseBlock):
                tool_usage[event.index] = AnthropicToolCall(
                    id=event.content_block.id,
                    name=event.content_block.name,
                )
                return StreamState(
                    terminal_state=False,
                )
            else:
                return StreamState(
                    terminal_state=False,
                )

        # corresponds to 529 error codes
        if event.type == StreamTypes.ERROR.value:
            logging.warning("Received error payload while streaming messages")
            return StreamState(
                terminal_state=True,
            )

        return None

    @staticmethod
    def _handle_streaming_delta_event(  # type: ignore
        event, chat: bool, tool_partials: Dict[int, List[str]]
    ):
        delta = (
            event.delta if event.type == StreamTypes.CONTENT_BLOCK_DELTA.value else None
        )
        delta_text = ""
        delta_thinking = ""
        if chat:
            # Anthropic Messages API stream does not return entire json argument,
            # it will give it in parts
            # https://docs.anthropic.com/en/api/messages-streaming#input-json-delta
            if isinstance(delta, InputJSONDelta):
                # building tool usage for content index with list
                # ensure the list exists before we try to append
                # the partial json string
                if event.index not in tool_partials:
                    tool_partials[event.index] = []

                tool_partials[event.index] += delta.partial_json
            if isinstance(delta, TextDelta):
                delta_text = event.delta.text
            if isinstance(delta, ThinkingDelta):
                delta_thinking = event.delta.thinking
        else:
            if isinstance(delta, TextDelta):
                delta_text = event.delta.text

        return AnthropicStreamDeltaEvent(
            delta=delta, thinking=delta_thinking, text=delta_text
        )

    @no_type_check
    def _process_stream_event(
        self,
        event,
        tool_usage: Dict[int, AnthropicToolCall],
        tool_partials: Dict[int, List[str]],
        chat: bool = False,
        completion: str = "",
        reasoning: str = "",
    ) -> StreamState:
        """
        Helper function to process SSE message while communicating with
        Anthropic's Message API with stream = True.
        """

        if early_exit := self._handle_early_stream_terminations(event, tool_usage):
            return early_exit

        processed_delta = self._handle_streaming_delta_event(event, chat, tool_partials)

        stop_reason = ""

        if isinstance(event, MessageDeltaEvent):
            stop_reason = event.delta.stop_reason

        if processed_delta.text:
            completion += processed_delta.text
            sys.stdout.write(Colors().GREEN + processed_delta.text)
            sys.stdout.flush()
            self.config.streamer(processed_delta.text, StreamEventType.TEXT)

        if processed_delta.thinking:
            reasoning += processed_delta.thinking
            sys.stdout.write(Colors().GREEN_DIM + processed_delta.thinking)
            sys.stdout.flush()
            self.config.streamer(processed_delta.thinking, StreamEventType.TEXT)

        # if we've hit the end of a content block, and we find the corresponding index
        # in our tool usage, we have a tool to call, and it's partial json should be
        # entirely present in tool_delta_partials
        if isinstance(event, ContentBlockStopEvent) and event.index in tool_usage:
            # try to grab tool by end block index
            tool_call = tool_usage.get(event.index)
            # log out the stored tool name
            sys.stdout.write(
                Colors().GREEN + "ANTHROPIC-TOOL: " + tool_call.name + ": "
            )
            sys.stdout.flush()
            self.config.streamer(tool_call.name, StreamEventType.TOOL_NAME)
            # generate the argument string from the partials
            argument = "".join(tool_partials[event.index])
            # only write if we have something to show
            if argument:
                sys.stdout.write(Colors().GREEN + argument)
                sys.stdout.flush()
                self.config.streamer(argument, StreamEventType.TOOL_ARGS)

        if stop_reason in ["tool_use", "stop_sequence"]:
            return StreamState(
                terminal_state=True,
            )

        return StreamState(
            terminal_state=False,
        )

    @no_type_check
    async def _process_stream_event_async(
        self,
        event,
        tool_usage: Dict[int, AnthropicToolCall],
        tool_partials: Dict[int, List[str]],
        chat: bool = False,
        completion: str = "",
        reasoning: str = "",
    ) -> StreamState:
        """
        Helper function to process SSE message while communicating with
        Anthropic's Message API with stream = True.
        """

        if early_exit := self._handle_early_stream_terminations(event, tool_usage):
            return early_exit

        silent = self.config.async_stream_quiet

        processed_delta = self._handle_streaming_delta_event(event, chat, tool_partials)

        stop_reason = ""

        if isinstance(event, MessageDeltaEvent):
            stop_reason = event.delta.stop_reason

        if processed_delta.text:
            completion += processed_delta.text
            if not silent:
                sys.stdout.write(Colors().GREEN + processed_delta.text)
                sys.stdout.flush()
                await self.config.streamer_async(
                    processed_delta.text, StreamEventType.TEXT
                )

        if processed_delta.thinking:
            reasoning += processed_delta.thinking
            if not silent:
                sys.stdout.write(Colors().GREEN_DIM + processed_delta.thinking)
                sys.stdout.flush()
                self.config.streamer_async(
                    processed_delta.thinking, StreamEventType.TEXT
                )

        if isinstance(event, ContentBlockStopEvent) and event.index in tool_usage:
            # try to grab tool by end block index
            tool_call = tool_usage.get(event.index)
            if not silent:
                # log out the stored tool name
                sys.stdout.write(
                    Colors().GREEN + "ANTHROPIC-TOOL: " + tool_call.name + ": "
                )
                sys.stdout.flush()
                await self.config.streamer_async(
                    tool_call.name, StreamEventType.TOOL_NAME
                )
            # generate the argument string from the partials
            argument = "".join(tool_partials[event.index])
            # only write if we have something to show
            if argument and not silent:
                sys.stdout.write(Colors().GREEN + argument)
                sys.stdout.flush()
                await self.config.streamer_async(argument, StreamEventType.TOOL_ARGS)

        if stop_reason in ["tool_use", "stop_sequence"]:
            return StreamState(
                terminal_state=True,
            )

        return StreamState(
            terminal_state=False,
        )

    @staticmethod
    def _create_stream_response(
        tool_usage: Dict[int, AnthropicToolCall],
        completion: str,
        reasoning: str,
    ) -> LLMResponse:
        """
        Create an LLMResponse object from the Messages API streaming response.
        """
        return LLMResponse(
            message=completion,
            reasoning=reasoning,
            cached=False,
            anthropic_tool_calls=list(tool_usage.values()) if tool_usage else None,
        )

    @retry_with_exponential_backoff(
        retryable_errors=retryable_anthropic_errors,
        terminal_errors=terminal_anthropic_errors,
    )
    def _completions_with_backoff(self, **kwargs) -> LLMResponse:  # type: ignore
        if self.client is None:
            raise ValueError("Anthropic chat-completion client is not set")
        completion_call = self.client.messages.create
        result = completion_call(**kwargs)
        if self.get_stream():
            stream_response: LLMResponse = self._stream_response(result)
            return stream_response
        if not isinstance(result, dict):
            result = result.dict()
        return self._process_chat_completion_response(result)

    def _hydrate_completion_call(
        self, completion_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        hydrated = {}
        if self.config.params:
            hydrated.update(self.config.params.to_dict_exclude_none())
        hydrated.update(completion_call)
        return hydrated
