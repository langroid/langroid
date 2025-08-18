from __future__ import annotations

import hashlib
import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic_settings import SettingsConfigDict

from langroid.cachedb.base import CacheDB
from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig

from .base import (
    LanguageModel,
    LLMConfig,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
    ToolChoiceTypes,
)
from .openai_gpt import OpenAIGPTConfig
from .utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class OpenAIResponsesConfig(OpenAIGPTConfig):
    """Configuration for the OpenAI Responses API provider.

    Inherits most behavior from OpenAIGPTConfig (auth, headers, http client knobs,
    retry/cache settings). Irrelevant fields (e.g., completion-specific) are ignored.
    """

    type: str = "openai_responses"
    reasoning_effort: str | None = None  # "low", "medium", or "high" for o1 models

    # Keep same env prefix behavior as OpenAIGPTConfig (OPENAI_*)
    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class OpenAIResponses(LanguageModel):
    """OpenAI Responses API provider implementing the `LanguageModel` interface.

    Stage 1 provided stubs; Stage 2 adds minimal non-stream chat via
    `client.responses.create` with basic request/response mapping and usage.
    Subsequent stages will add streaming, tools, structured output, vision, etc.
    """

    @property
    def supports_strict_tools(self) -> bool:
        """Check if this model supports strict tool schemas."""
        # Check model capabilities - most modern OpenAI models support this
        model = self.config.chat_model.lower()
        return "gpt-4" in model or "gpt-3.5" in model or "o1" in model

    @property
    def supports_json_schema(self) -> bool:
        """Check if this model supports JSON schema output format."""
        model = self.config.chat_model.lower()
        return "gpt-4" in model or "gpt-3.5" in model or "o1" in model

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize OpenAI Responses API client."""
        # Accept either OpenAIGPTConfig or OpenAIResponsesConfig
        if config is None:
            config = OpenAIResponsesConfig()
        elif hasattr(config, "use_responses_api"):  # It's an OpenAIGPTConfig
            from langroid.language_models.openai_gpt import OpenAIGPTConfig

            if isinstance(config, OpenAIGPTConfig) and not isinstance(
                config, OpenAIResponsesConfig
            ):
                # Convert OpenAIGPTConfig to OpenAIResponsesConfig
                responses_config = OpenAIResponsesConfig(**config.model_dump())
                config = responses_config

        super().__init__(config)
        self.config: OpenAIResponsesConfig = config  # type: ignore

        # Initialize cache if configured
        self.cache: Optional[CacheDB] = None
        if self.config.cache_config is not None:
            if isinstance(self.config.cache_config, RedisCacheConfig):
                self.cache = RedisCache(self.config.cache_config)
            else:
                raise ValueError(
                    f"Unsupported cache type: {type(self.config.cache_config)}"
                )

    def _convert_tool_spec(self, tool: OpenAIToolSpec) -> Dict[str, Any]:
        """Convert OpenAIToolSpec to Responses API format."""
        # For Responses API, tools are in the same format as Chat Completions
        return tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)

    def _messages_to_input_parts(
        self, messages: List[LLMMessage]
    ) -> List[Dict[str, Any]]:
        """Convert messages to Responses API input parts."""
        from langroid.language_models.base import Role

        input_parts: List[Dict[str, Any]] = []

        # Process messages in order
        for msg in messages:
            if msg.role == Role.USER:
                # User messages become input_text parts
                input_parts.append({"type": "input_text", "text": msg.content})

                # Add any file attachments as image parts
                if msg.files:
                    for file_attachment in msg.files:
                        image_part = self._convert_file_attachment(file_attachment)
                        if image_part:
                            input_parts.append(image_part)

            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant tool calls - these are already in the conversation
                # We don't add them as input parts, they're part of history
                pass
            elif msg.role == Role.TOOL:
                # Tool results become tool_result parts
                input_parts.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": msg.tool_call_id,
                        "output": msg.content,
                    }
                )

        # If no user content was added, add from last user message
        if not any(p.get("type") == "input_text" for p in input_parts):
            user_msgs = [m for m in messages if m.role == Role.USER]
            if user_msgs:
                last_user = user_msgs[-1]
                input_parts.append({"type": "input_text", "text": last_user.content})
                # Also add files from last user message if any
                if last_user.files:
                    for file_attachment in last_user.files:
                        image_part = self._convert_file_attachment(file_attachment)
                        if image_part:
                            input_parts.append(image_part)
            else:
                # Fallback
                input_parts.append({"type": "input_text", "text": "Hello"})

        return input_parts

    def _convert_file_attachment(self, attachment: Any) -> Optional[Dict[str, Any]]:
        """Convert a FileAttachment to Responses API image format."""
        from langroid.parsing.file_attachment import FileAttachment

        if not isinstance(attachment, FileAttachment):
            return None

        # Get the URL (could be data URI or HTTP URL)
        url = attachment.url
        if not url:
            return None

        # Create image part for Responses API
        return {
            "type": "image",
            "image": url,  # Responses API accepts both data URIs and HTTP URLs
        }

    def set_stream(self, stream: bool) -> bool:  # pragma: no cover - trivial
        prev = self.config.stream
        self.config.stream = stream
        return prev

    def get_stream(self) -> bool:  # pragma: no cover - trivial
        return self.config.stream

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        raise NotImplementedError("OpenAIResponses.generate not implemented yet")

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        raise NotImplementedError("OpenAIResponses.agenerate not implemented yet")

    def _cache_lookup(self, fn_name: str, **kwargs: Any) -> Tuple[str, Any]:
        """Look up cached result for given function and arguments.

        Returns:
            Tuple of (cache_key, cached_result or None)
        """
        if self.cache is None:
            return "", None

        # Check global cache setting
        from langroid import settings  # type: ignore

        if not settings.cache:
            return "", None

        # Create deterministic cache key from function name and kwargs
        sorted_kwargs_str = str(sorted(kwargs.items()))
        raw_key = f"{fn_name}:{sorted_kwargs_str}"

        # Hash the key to fixed length using SHA256
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

        # Look up in cache
        cached_result = self.cache.retrieve(hashed_key)
        return hashed_key, cached_result

    def _cache_store(self, key: str, value: Any) -> None:
        """Store a value in cache with given key."""
        if self.cache is None:
            return

        # Check global cache setting
        from langroid import settings  # type: ignore

        if not settings.cache:
            return
        try:
            self.cache.store(key, value)
        except Exception as e:
            logger.error(f"Error storing cache: {e}")

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        # Minimal non-stream chat implementation (Stage 2)
        from langroid.language_models.base import LLMTokenUsage, Role
        from langroid.language_models.client_cache import get_openai_client
        from langroid.language_models.model_info import get_model_info

        # Handle edge cases for messages input
        if isinstance(messages, str):
            # Wrap string into simple system+user dialog
            messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages if messages else "Hello"),
            ]
        elif not messages:
            # Empty message list - create default
            messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content="Hello"),
            ]

        # Check if this is an o1 reasoning model
        is_o1_model = "o1" in self.config.chat_model.lower()

        # Build instructions from SYSTEM messages and a simple user input block
        instructions_parts: List[str] = [
            m.content for m in messages if m.role == Role.SYSTEM
        ]

        # For o1 models, convert system messages to user messages
        if is_o1_model and instructions_parts:
            # o1 models don't support system messages, merge them into user content
            system_context = "\n\n".join(instructions_parts)
            # Find first user message and prepend system context
            for msg in messages:
                if msg.role == Role.USER:
                    msg.content = f"{system_context}\n\n{msg.content}"
                    break
            instructions = None  # No separate instructions for o1
        else:
            instructions = (
                "\n\n".join(instructions_parts) if len(instructions_parts) > 0 else None
            )

        # Collect the last USER message content and attachments (minimal for Stage 2)
        user_msgs = [m for m in messages if m.role == Role.USER]
        if len(user_msgs) == 0:
            # Fallback: create a placeholder user message if none was provided
            user_msgs = [LLMMessage(role=Role.USER, content="Hello")]

        # Map messages to input parts for Responses API
        input_parts = self._messages_to_input_parts(messages)

        # Compute a conservative max_output_tokens honoring method arg over config
        max_output_tokens = (
            max_tokens
            if max_tokens is not None
            else self.config.model_max_output_tokens
        )

        # Prepare headers - add beta header if using strict JSON schema
        headers = dict(self.config.headers) if self.config.headers else {}
        if response_format is not None:
            # Check if strict mode is enabled
            if hasattr(response_format, "strict") and response_format.strict:
                headers["openai-beta"] = "structured-outputs-v1"
            elif isinstance(response_format, dict) and response_format.get("strict"):
                headers["openai-beta"] = "structured-outputs-v1"

        # Create a client (reusing cached client if configured)
        client = get_openai_client(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            default_headers=headers,
        )

        # Prepare request payload
        req: Dict[str, Any] = {
            "model": self.config.chat_model,
            "input": [
                {
                    "role": "user",
                    "content": input_parts,
                }
            ],
            "max_output_tokens": max_output_tokens,
            "temperature": (
                1.0 if is_o1_model else self.config.temperature
            ),  # o1 always uses temperature=1
        }
        if instructions:
            req["instructions"] = instructions

        # Add reasoning_effort for o1 models
        if is_o1_model and self.config.reasoning_effort:
            req["reasoning_effort"] = self.config.reasoning_effort

        # Add tools if provided (but not for o1 models)
        if tools and not is_o1_model:
            req["tools"] = [self._convert_tool_spec(t) for t in tools]
            # Handle tool_choice parameter
            if tool_choice == "none":
                req["tool_choice"] = "none"
            elif tool_choice == "required":
                req["tool_choice"] = "required"
            elif tool_choice == "auto":
                req["tool_choice"] = "auto"
            elif isinstance(tool_choice, dict) and "function" in tool_choice:
                # Specific tool choice
                func_dict = tool_choice.get("function", {})
                if isinstance(func_dict, dict):
                    req["tool_choice"] = {
                        "type": "function",
                        "function": {"name": func_dict.get("name", "")},
                    }

        # Add response_format if provided
        if response_format is not None:
            # Convert OpenAIJsonSchemaSpec to API format
            if hasattr(response_format, "to_dict"):
                req["response_format"] = response_format.to_dict()
            elif hasattr(response_format, "type"):
                # Simple format like {"type": "json_object"}
                req["response_format"] = {"type": response_format.type}
            else:
                # Direct dict format
                req["response_format"] = response_format

        # Check cache before making API call
        cache_key, cached_result = self._cache_lookup(
            "ResponsesChat",
            model=self.config.chat_model,
            messages=str(messages),
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            tools=str(tools) if tools else None,
            tool_choice=str(tool_choice),
            response_format=str(response_format) if response_format else None,
            stream=self.config.stream,
        )

        if cached_result is not None:
            # Return cached response
            from langroid.language_models.base import LLMResponse, LLMTokenUsage

            return LLMResponse(
                message=cached_result.get("message", ""),
                reasoning=cached_result.get("reasoning", ""),
                usage=LLMTokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    cached_tokens=cached_result.get("total_tokens", 0),
                    cost=0,  # No cost for cached responses
                ),
                oai_tool_calls=cached_result.get("oai_tool_calls"),
                cached=True,
            )

        # Check if we should stream and if Responses API is available
        use_responses_api = hasattr(client, "responses") and callable(
            getattr(client, "responses").create  # type: ignore[attr-defined]
        )

        # Create retry wrapper with config params
        retry_decorator = partial(
            retry_with_exponential_backoff,
            initial_delay=self.config.retry_params.initial_delay,
            exponential_base=self.config.retry_params.exponential_base,
            jitter=self.config.retry_params.jitter,
            max_retries=self.config.retry_params.max_retries,
        )

        @retry_decorator
        def _api_call_with_retry(**kwargs: Any) -> Any:
            """Make API call with retry logic."""
            try:
                if use_responses_api and not kwargs.get("use_chat_completions"):
                    return client.responses.create(**kwargs)  # type: ignore[attr-defined]
                else:
                    # Use Chat Completions API
                    return client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                raise

        if self.config.stream and use_responses_api:
            # Stage 3: Streaming support for Responses API
            return self._stream_response(client, req, cache_key)
        elif use_responses_api:
            # Non-streaming Responses API call with retry
            try:
                result = _api_call_with_retry(**req)
            except Exception as e:
                logger.warning(
                    f"Responses API failed, falling back to Chat Completions: {e}"
                )
                # Fall back to Chat Completions on Responses API failure
                use_responses_api = False
                # Continue to Chat Completions fallback below

        if not use_responses_api:
            # Fallback: emulate via Chat Completions
            # Map messages to OpenAI chat format
            chat_messages: List[Dict[str, Any]] = []
            if instructions:
                chat_messages.append({"role": "system", "content": instructions})

            # Convert all messages for Chat Completions
            for msg in messages:
                if msg.role == Role.USER:
                    # Handle user messages with potential file attachments
                    if msg.files and len(msg.files) > 0:
                        # Create content array with text and images
                        content_parts: List[Dict[str, Any]] = [
                            {"type": "text", "text": msg.content}
                        ]
                        for file_attachment in msg.files:
                            if file_attachment.url:
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": file_attachment.url},
                                    }
                                )
                        chat_messages.append({"role": "user", "content": content_parts})
                    else:
                        chat_messages.append({"role": "user", "content": msg.content})
                elif msg.role == Role.ASSISTANT:
                    chat_msg: Dict[str, Any] = {"role": "assistant"}
                    if msg.content:
                        chat_msg["content"] = msg.content
                    if msg.tool_calls:
                        chat_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name if tc.function else "",
                                    "arguments": (
                                        tc.function.arguments
                                        if tc.function
                                        and isinstance(tc.function.arguments, str)
                                        else json.dumps(
                                            tc.function.arguments if tc.function else {}
                                        )
                                    ),
                                },
                            }
                            for tc in msg.tool_calls
                        ]
                    chat_messages.append(chat_msg)
                elif msg.role == Role.TOOL:
                    chat_messages.append(
                        {
                            "role": "tool",
                            "content": msg.content,
                            "tool_call_id": msg.tool_call_id,
                        }
                    )

            cc_req: Dict[str, Any] = {
                "model": self.config.chat_model,
                "messages": chat_messages,
                "max_tokens": max_output_tokens,
                "temperature": self.config.temperature,
                "stream": self.config.stream,
            }

            # Add tools to Chat Completions request if provided
            if tools:
                cc_req["tools"] = [self._convert_tool_spec(t) for t in tools]
                if tool_choice:
                    cc_req["tool_choice"] = tool_choice

            # Add response_format to Chat Completions request if provided
            if response_format is not None:
                if hasattr(response_format, "to_dict"):
                    cc_req["response_format"] = response_format.to_dict()
                elif hasattr(response_format, "type"):
                    cc_req["response_format"] = {"type": response_format.type}
                else:
                    cc_req["response_format"] = response_format

            if self.config.stream:
                # Streaming Chat Completions fallback
                return self._stream_chat_completions(client, cc_req, cache_key)
            else:
                # Non-streaming Chat Completions with retry
                result = _api_call_with_retry(**cc_req, use_chat_completions=True)

        # Extract message text, reasoning, and tool calls from response
        message_text = ""
        reasoning_text = ""
        tool_calls = None
        usage_dict: Dict[str, Any] = {}
        if use_responses_api:
            try:
                # SDK object may have attributes
                message_text = getattr(result, "output_text", "") or ""
                # Extract reasoning for o1 models
                reasoning_text = getattr(result, "reasoning", "") or ""
                usage_obj = getattr(result, "usage", None)
                if usage_obj is not None:
                    usage_dict = (
                        usage_obj.model_dump()
                        if hasattr(usage_obj, "model_dump")
                        else dict(usage_obj)
                    )
            except Exception:
                pass

            # Try to extract tool calls
            if hasattr(result, "output") and result.output:
                for item in result.output:
                    if hasattr(item, "type") and item.type == "tool_call":
                        if tool_calls is None:
                            tool_calls = []
                        # Convert to OpenAIToolCall format
                        from langroid.language_models.base import (
                            LLMFunctionCall,
                            OpenAIToolCall,
                        )

                        tool_calls.append(
                            OpenAIToolCall(
                                id=getattr(item, "id", None),
                                type="function",
                                function=LLMFunctionCall(
                                    name=(
                                        item.function.name
                                        if hasattr(item, "function")
                                        else ""
                                    ),
                                    arguments=(
                                        item.function.arguments
                                        if hasattr(item, "function")
                                        else "{}"
                                    ),
                                ),
                            )
                        )

            if not message_text:
                # Fallback to dict inspection
                rd = (
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else (result if isinstance(result, dict) else {})
                )
                message_text = rd.get("output_text", "") or ""
                if not message_text:
                    # Try first text part in output
                    out = rd.get("output", [])
                    if out:
                        # Find first message entry with text or tool calls
                        for entry in out:
                            if entry.get("type") == "message":
                                parts = entry.get("content", [])
                                for p in parts:
                                    # Responses: output_text or text type
                                    if p.get("type") in (
                                        "output_text",
                                        "text",
                                    ) and p.get("text"):
                                        message_text = p["text"]
                                        break
                            elif entry.get("type") == "tool_call":
                                # Extract tool call
                                if tool_calls is None:
                                    tool_calls = []
                                from langroid.language_models.base import (
                                    LLMFunctionCall,
                                    OpenAIToolCall,
                                )

                                tool_calls.append(
                                    OpenAIToolCall(
                                        id=entry.get("id"),
                                        type="function",
                                        function=LLMFunctionCall(
                                            name=entry.get("function", {}).get(
                                                "name", ""
                                            ),
                                            arguments=entry.get("function", {}).get(
                                                "arguments", "{}"
                                            ),
                                        ),
                                    )
                                )
                            if message_text:
                                break
                usage_dict = rd.get("usage", {}) or usage_dict
        else:
            # Chat Completions result mapping
            rd = result.model_dump() if hasattr(result, "model_dump") else result
            if rd and "choices" in rd and rd["choices"]:
                choice0 = rd["choices"][0]
                if "message" in choice0 and choice0["message"]:
                    message_text = (choice0["message"].get("content") or "").strip()
                    # Extract tool calls from Chat Completions response
                    if choice0["message"].get("tool_calls"):
                        from langroid.language_models.base import (
                            LLMFunctionCall,
                            OpenAIToolCall,
                        )

                        tool_calls = []
                        for tc in choice0["message"]["tool_calls"]:
                            tool_calls.append(
                                OpenAIToolCall(
                                    id=tc.get("id"),
                                    type=tc.get("type", "function"),
                                    function=LLMFunctionCall(
                                        name=tc.get("function", {}).get("name", ""),
                                        arguments=tc.get("function", {}).get(
                                            "arguments", "{}"
                                        ),
                                    ),
                                )
                            )
                elif "text" in choice0:
                    message_text = (choice0.get("text") or "").strip()
            usage_dict = rd.get("usage", {}) if isinstance(rd, dict) else {}

        # Build usage, cost
        # Usage mapping differs between Responses and Chat Completions
        if use_responses_api:
            prompt_tokens = int(usage_dict.get("input_tokens", 0))
            completion_tokens = int(usage_dict.get("output_tokens", 0))
            cached_tokens = int(usage_dict.get("cached_tokens", 0))
        else:
            prompt_tokens = int(usage_dict.get("prompt_tokens", 0))
            completion_tokens = int(usage_dict.get("completion_tokens", 0))
            # Chat Completions API may report cached_prompt_tokens
            cached_tokens = int(usage_dict.get("cached_prompt_tokens", 0))

        # Compute cost using model info
        info = get_model_info(self.config.chat_model)
        # Price per 1K tokens for chat models
        input_per_k = info.input_cost_per_million / 1000.0
        cached_per_k = (
            info.cached_cost_per_million or info.input_cost_per_million
        ) / 1000.0
        output_per_k = info.output_cost_per_million / 1000.0
        cost = (
            input_per_k * (prompt_tokens - cached_tokens) / 1000.0
            + cached_per_k * cached_tokens / 1000.0
            + output_per_k * completion_tokens / 1000.0
        )

        self.update_usage_cost(
            chat=True, prompts=prompt_tokens, completions=completion_tokens, cost=cost
        )

        from langroid.language_models.base import LLMResponse

        usage = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )

        response = LLMResponse(
            message=message_text.strip(),
            reasoning=reasoning_text,
            usage=usage,
            oai_tool_calls=tool_calls,
        )

        # Store in cache if we have a cache key
        if cache_key and self.cache is not None:
            cache_value = {
                "message": response.message,
                "reasoning": response.reasoning,
                "total_tokens": usage.total_tokens,
                "oai_tool_calls": response.oai_tool_calls,
            }
            self._cache_store(cache_key, cache_value)

        return response

    def _stream_response(
        self, client: Any, request_params: Dict[str, Any], cache_key: str = ""
    ) -> LLMResponse:
        """Handle streaming response for Responses API."""
        from langroid.language_models.base import LLMResponse, LLMTokenUsage
        from langroid.language_models.model_info import get_model_info

        accumulated_text: List[str] = []
        accumulated_reasoning: List[str] = []  # Track reasoning deltas
        tool_calls: Dict[str, Any] = {}  # Track tool calls by ID
        final_response = None

        try:
            # Create streaming request with retry wrapper
            retry_decorator = partial(
                retry_with_exponential_backoff,
                initial_delay=self.config.retry_params.initial_delay,
                exponential_base=self.config.retry_params.exponential_base,
                jitter=self.config.retry_params.jitter,
                max_retries=self.config.retry_params.max_retries,
            )

            @retry_decorator
            def _create_stream() -> Any:
                return client.responses.create(**request_params, stream=True)  # type: ignore[attr-defined]

            stream = _create_stream()

            # Process stream events
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "response.output_text.delta":
                        # Text delta event
                        delta_text = getattr(event, "delta", "")
                        if delta_text:
                            accumulated_text.append(delta_text)
                            # Call streamer callback if configured
                            if self.config.streamer:
                                self.config.streamer(delta_text)

                    elif event.type == "response.reasoning.delta":
                        # Reasoning delta event for o1 models
                        delta_reasoning = getattr(event, "delta", "")
                        if delta_reasoning:
                            accumulated_reasoning.append(delta_reasoning)

                    elif event.type == "response.tool_call.delta":
                        # Tool call delta - accumulate tool calls
                        if hasattr(event, "delta"):
                            tool_id = getattr(event.delta, "id", None)
                            if tool_id:
                                if tool_id not in tool_calls:
                                    tool_calls[tool_id] = {
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                # Update function details
                                if hasattr(event.delta, "function"):
                                    func = event.delta.function
                                    if hasattr(func, "name") and func.name:
                                        tool_calls[tool_id]["function"][
                                            "name"
                                        ] = func.name
                                    if hasattr(func, "arguments") and func.arguments:
                                        tool_calls[tool_id]["function"][
                                            "arguments"
                                        ] += func.arguments

                    elif event.type == "response.completed":
                        # Final response with usage
                        final_response = event
                        break

                    elif event.type == "response.done":
                        # Alternative completion event
                        final_response = event
                        break
        except Exception as e:
            # Handle streaming errors gracefully
            import logging

            logging.warning(f"Streaming error: {e}")

        # Join accumulated text and reasoning
        message_text = "".join(accumulated_text).strip()
        reasoning_text = "".join(accumulated_reasoning).strip()

        # Extract usage from final response
        usage_dict: Dict[str, Any] = {}
        if final_response:
            if hasattr(final_response, "usage"):
                usage_obj = final_response.usage
                usage_dict = (
                    usage_obj.model_dump()
                    if hasattr(usage_obj, "model_dump")
                    else dict(usage_obj)
                )
            elif hasattr(final_response, "data") and hasattr(
                final_response.data, "usage"
            ):
                usage_obj = final_response.data.usage
                usage_dict = (
                    usage_obj.model_dump()
                    if hasattr(usage_obj, "model_dump")
                    else dict(usage_obj)
                )

        # Build usage with Responses API field names
        prompt_tokens = int(usage_dict.get("input_tokens", 0))
        completion_tokens = int(usage_dict.get("output_tokens", 0))
        cached_tokens = int(usage_dict.get("cached_tokens", 0))

        # Compute cost
        info = get_model_info(self.config.chat_model)
        input_per_k = info.input_cost_per_million / 1000.0
        cached_per_k = (
            info.cached_cost_per_million or info.input_cost_per_million
        ) / 1000.0
        output_per_k = info.output_cost_per_million / 1000.0
        cost = (
            input_per_k * (prompt_tokens - cached_tokens) / 1000.0
            + cached_per_k * cached_tokens / 1000.0
            + output_per_k * completion_tokens / 1000.0
        )

        self.update_usage_cost(
            chat=True, prompts=prompt_tokens, completions=completion_tokens, cost=cost
        )

        usage = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )

        # Convert tool_calls dict to list of OpenAIToolCall objects
        oai_tool_calls = None
        if tool_calls:
            from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall

            oai_tool_calls = []
            for tc in tool_calls.values():
                oai_tool_calls.append(
                    OpenAIToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=LLMFunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

        response = LLMResponse(
            message=message_text,
            reasoning=reasoning_text,
            usage=usage,
            oai_tool_calls=oai_tool_calls,
        )

        # Store streaming response in cache after completion
        if cache_key and self.cache is not None:
            cache_value = {
                "message": response.message,
                "reasoning": response.reasoning,
                "total_tokens": usage.total_tokens,
                "oai_tool_calls": response.oai_tool_calls,
            }
            self._cache_store(cache_key, cache_value)

        return response

    def _stream_chat_completions(
        self, client: Any, request_params: Dict[str, Any], cache_key: str = ""
    ) -> LLMResponse:
        """Handle streaming response for Chat Completions API fallback."""
        from langroid.language_models.base import LLMResponse, LLMTokenUsage
        from langroid.language_models.model_info import get_model_info

        accumulated_text: List[str] = []
        reasoning_text = ""  # Chat Completions API doesn't return reasoning
        tool_calls: Dict[str, Any] = {}  # Track tool calls by index
        usage_dict: Dict[str, Any] = {}

        try:
            # Create streaming request with retry wrapper
            retry_decorator = partial(
                retry_with_exponential_backoff,
                initial_delay=self.config.retry_params.initial_delay,
                exponential_base=self.config.retry_params.exponential_base,
                jitter=self.config.retry_params.jitter,
                max_retries=self.config.retry_params.max_retries,
            )

            @retry_decorator
            def _create_stream() -> Any:
                return client.chat.completions.create(**request_params)

            stream = _create_stream()

            # Process stream chunks
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta"):
                        # Handle text content
                        if hasattr(choice.delta, "content"):
                            delta_text = choice.delta.content
                            if delta_text:
                                accumulated_text.append(delta_text)
                                # Call streamer callback if configured
                                if self.config.streamer:
                                    self.config.streamer(delta_text)

                        # Handle tool calls
                        if (
                            hasattr(choice.delta, "tool_calls")
                            and choice.delta.tool_calls
                        ):
                            for tc_delta in choice.delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls:
                                    tool_calls[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if hasattr(tc_delta, "id") and tc_delta.id:
                                    tool_calls[idx]["id"] = tc_delta.id
                                if hasattr(tc_delta, "function"):
                                    if (
                                        hasattr(tc_delta.function, "name")
                                        and tc_delta.function.name
                                    ):
                                        tool_calls[idx]["function"][
                                            "name"
                                        ] = tc_delta.function.name
                                    if (
                                        hasattr(tc_delta.function, "arguments")
                                        and tc_delta.function.arguments
                                    ):
                                        tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc_delta.function.arguments

                # Some models include usage in final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage
                    usage_dict = (
                        usage_obj.model_dump()
                        if hasattr(usage_obj, "model_dump")
                        else dict(usage_obj)
                    )
        except Exception as e:
            # Handle streaming errors gracefully
            import logging

            logging.warning(f"Chat Completions streaming error: {e}")

        # Join accumulated text
        message_text = "".join(accumulated_text).strip()

        # Build usage with Chat Completions API field names
        prompt_tokens = int(usage_dict.get("prompt_tokens", 0))
        completion_tokens = int(usage_dict.get("completion_tokens", 0))
        # Chat Completions API may report cached_prompt_tokens in usage
        cached_tokens = int(usage_dict.get("cached_prompt_tokens", 0))

        # Compute cost
        info = get_model_info(self.config.chat_model)
        input_per_k = info.input_cost_per_million / 1000.0
        cached_per_k = (
            info.cached_cost_per_million or info.input_cost_per_million
        ) / 1000.0
        output_per_k = info.output_cost_per_million / 1000.0
        cost = (
            input_per_k * (prompt_tokens - cached_tokens) / 1000.0
            + cached_per_k * cached_tokens / 1000.0
            + output_per_k * completion_tokens / 1000.0
        )

        self.update_usage_cost(
            chat=True, prompts=prompt_tokens, completions=completion_tokens, cost=cost
        )

        usage = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )

        # Convert tool_calls dict to list of OpenAIToolCall objects
        oai_tool_calls = None
        if tool_calls:
            from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall

            oai_tool_calls = []
            for tc in tool_calls.values():
                oai_tool_calls.append(
                    OpenAIToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=LLMFunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

        response = LLMResponse(
            message=message_text,
            reasoning=reasoning_text,
            usage=usage,
            oai_tool_calls=oai_tool_calls,
        )

        # Store streaming response in cache after completion
        if cache_key and self.cache is not None:
            cache_value = {
                "message": response.message,
                "reasoning": response.reasoning,
                "total_tokens": usage.total_tokens,
                "oai_tool_calls": response.oai_tool_calls,
            }
            self._cache_store(cache_key, cache_value)

        return response

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        raise NotImplementedError("OpenAIResponses.achat not implemented yet")
