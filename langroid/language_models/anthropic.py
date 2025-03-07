import logging
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union, no_type_check

import anthropic.types
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
from anthropic.types import RawMessageStreamEvent
from rich import print

from langroid.language_models import (
    LLMConfig,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    StreamEventType,
)
from langroid.language_models.base import (
    LanguageModel,
    LLMCallConfig,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
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
default_anthropic_completion_model = filter_default_model(
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


class AnthropicSystemCacheControl(BaseModel):
    type: str = "ephemeral"


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


class AnthropicToolSpec(BaseModel):
    """
    Class defining an available tool
    that Anthropic can potentially leverage.
    https://docs.anthropic.com/en/api/messages#body-tools
    """

    name: str
    # json object
    input_schema: Dict[str, Any]
    # Strongly recommended to fill
    description: str | None = ""
    cache_control: AnthropicSystemCacheControl | None = None
    type: str | None = "custom"


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


class SynchronousStreamEnded(BaseModel):
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
    completion_model = default_anthropic_completion_model
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

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        return LLMResponse(message="TODO")

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        return LLMResponse(message="TODO")

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        tool_variants: ToolVariantSelector = ToolVariantSelector(
            open_ai=[], anthropic=[]
        ),
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "",
        response_format: OpenAIJsonSchemaSpec | None = None,
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
        """

        if functions:
            logging.warning("Functions are unavailable with Anthropic's Messages API")

        if function_call:
            logging.warning(
                "Function calls are unavailable with Anthropic's Messages API"
            )

        if functions or function_call:
            logging.warning(
                """You might observe an unexpected answer with function usage
                for Anthropic Message calls."""
            )
            logging.warning("Please use the tools and tool_choice parameters instead.")

        if self.config.use_completion_for_chat:
            # upcoming generate function, subsequent commits
            pass

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
    ) -> LLMResponse:
        return LLMResponse(message="TODO")

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
        # we need to check if its a stream response
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

    @no_type_check
    def _process_stream_event(
        self,
        event: RawMessageStreamEvent,
        tool_usage: Dict[int, AnthropicToolCall],
        tool_partials: Dict[int, List[str]],
        chat: bool = False,
        completion: str = "",
        reasoning: str = "",
    ) -> SynchronousStreamEnded:
        """
        Helper function to process SSE message while communicating with
        Anthropic's Message API with stream = True.
        """

        if event.type in [
            StreamTypes.PING.value,
            StreamTypes.MESSAGE_START.value,
        ]:
            return SynchronousStreamEnded(
                terminal_state=False,
            )

        # we need a different check for content_block_start
        # because Anthropic can have a tool_use call here,
        # and this is the response they place metadata for tool
        # usage such as tool_id and tool_name
        if event.type == StreamTypes.CONTENT_BLOCK_START.value:
            if isinstance(event.content_block, anthropic.types.ToolUseBlock):
                tool_usage[event.index] = AnthropicToolCall(
                    id=event.content_block.id,
                    name=event.content_block.name,
                )
                return SynchronousStreamEnded(
                    terminal_state=False,
                )
            else:
                return SynchronousStreamEnded(
                    terminal_state=False,
                )

        # corresponds to 529 error codes
        if event.type == StreamTypes.ERROR.value:
            logging.warning("Received error payload while streaming messages")
            return SynchronousStreamEnded(
                terminal_state=True,
            )

        delta = (
            event.delta if event.type == StreamTypes.CONTENT_BLOCK_DELTA.value else None
        )

        delta_text = ""
        delta_thinking = ""

        if chat:
            # Anthropic Messages API stream does not return entire json argument,
            # it will give it in parts
            # https://docs.anthropic.com/en/api/messages-streaming#input-json-delta
            if isinstance(delta, anthropic.types.InputJSONDelta):
                # building tool usage for content index with list
                # ensure the list exists before we try to append
                # the partial json string
                if event.index not in tool_partials:
                    tool_partials[event.index] = []

                tool_partials[event.index] += delta.partial_json
            if isinstance(delta, anthropic.types.TextDelta):
                delta_text = event.delta.text
            if isinstance(delta, anthropic.types.ThinkingDelta):
                delta_thinking = event.delta.thinking
        else:
            if isinstance(delta, anthropic.types.TextDelta):
                delta_text = event.delta.text

        stop_reason = ""

        if isinstance(event, anthropic.types.MessageDeltaEvent):
            stop_reason = event.delta.stop_reason

        if delta_text:
            completion += delta_text
            sys.stdout.write(Colors().GREEN + delta_text)
            sys.stdout.flush()
            self.config.streamer(delta_text, StreamEventType.TEXT)

        if delta_thinking:
            reasoning += delta_thinking
            sys.stdout.write(Colors().GREEN_DIM + delta_thinking)
            sys.stdout.flush()
            self.config.streamer(delta_thinking, StreamEventType.TEXT)

        # if we've hit the end of a content block, and we find the corresponding index
        # in our tool usage, we have a tool to call, and it's partial json should be
        # entirely present in tool_delta_partials
        if (
            isinstance(event, anthropic.types.ContentBlockStopEvent)
            and event.index in tool_usage
        ):
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
            return SynchronousStreamEnded(
                terminal_state=True,
            )

        return SynchronousStreamEnded(
            terminal_state=False,
        )

    def _create_stream_response(
        self,
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
