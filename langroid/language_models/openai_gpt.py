import ast
import hashlib
import logging
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from pydantic import BaseModel
from rich import print

from langroid.cachedb.momento_cachedb import MomentoCache, MomentoCacheConfig
from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMFunctionCall,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    Role,
)
from langroid.language_models.prompt_formatter.base import (
    PromptFormatter,
)
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)
from langroid.utils.configuration import settings
from langroid.utils.constants import NO_ANSWER, Colors

logging.getLogger("openai").setLevel(logging.ERROR)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT3_5_TURBO = "gpt-3.5-turbo-0613"
    GPT4_NOFUNC = "gpt-4"  # before function_call API
    GPT4 = "gpt-4"
    LOCAL = "local"  # dummy for any local model


class OpenAICompletionModel(str, Enum):
    """Enum for OpenAI Completion models"""

    TEXT_DA_VINCI_003 = "text-davinci-003"
    TEXT_ADA_001 = "text-ada-001"
    GPT4 = "gpt-4"
    LOCAL = "local"  # dummy for any local model


class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    api_base: str | None = None  # used for local or other non-OpenAI models
    max_output_tokens: int = 1024
    min_output_tokens: int = 64
    timeout: int = 20
    temperature: float = 0.2
    chat_model: str | OpenAIChatModel = OpenAIChatModel.GPT4
    completion_model: str | OpenAICompletionModel = OpenAICompletionModel.GPT4
    context_length: Dict[str, int] = {
        OpenAIChatModel.GPT3_5_TURBO: 4096,
        OpenAIChatModel.GPT4: 8192,
        OpenAIChatModel.GPT4_NOFUNC: 8192,
        OpenAICompletionModel.TEXT_DA_VINCI_003: 4096,
    }
    cost_per_1k_tokens: Dict[str, Tuple[float, float]] = {
        # (input/prompt cost, output/completion cost)
        OpenAIChatModel.GPT3_5_TURBO: (0.0015, 0.002),
        OpenAIChatModel.GPT4: (0.03, 0.06),  # 8K context
        OpenAIChatModel.GPT4_NOFUNC: (0.03, 0.06),
    }

    # all of the non-dict vars above can be set via env vars,
    # by upper-casing the name and prefixing with OPENAI_, e.g.
    # OPENAI_MAX_OUTPUT_TOKENS=1000.
    # The dict fields can also be set via env vars by passing json strings, e.g.
    # OPENAI_CONTEXT_LENGTH='{"gpt-3.5-turbo-0613": 4096, "local": 8192}'
    class Config:
        env_prefix = "OPENAI_"


class OpenAIResponse(BaseModel):
    """OpenAI response model, either completion or chat."""

    choices: List[Dict]  # type: ignore
    usage: Dict  # type: ignore


# Define a class for OpenAI GPT-3 that extends the base class
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(self, config: OpenAIGPTConfig):
        """
        Args:
            config: configuration for openai-gpt model
        """
        super().__init__(config)
        if settings.nofunc:
            self.chat_model = OpenAIChatModel.GPT4_NOFUNC
        self.api_base: str | None = None
        if config.local:
            self.config.chat_model = config.local.model
            self.config.use_completion_for_chat = config.local.use_completion_for_chat
            self.config.use_chat_for_completion = config.local.use_chat_for_completion
            self.api_key = "sx-xxx"
            self.api_base = config.local.api_base
            config.context_length = {config.local.model: config.local.context_length}
            config.cost_per_1k_tokens = {
                config.local.model: (0.0, 0.0),
            }
        else:
            # TODO: get rid of this and add `api_key` to the OpenAIGPTConfig
            # so we can get it from the OPENAI_API_KEY env var
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        if self.api_key == "":
            raise ValueError(
                """
                OPENAI_API_KEY not set in .env file,
                please set it to your OpenAI API key."""
            )
        self.cache: MomentoCache | RedisCache
        if settings.cache_type == "momento":
            config.cache_config = MomentoCacheConfig()
            self.cache = MomentoCache(config.cache_config)
        else:
            config.cache_config = RedisCacheConfig()
            self.cache = RedisCache(config.cache_config)

    def set_stream(self, stream: bool) -> bool:
        """Enable or disable streaming output from API.
        Args:
            stream: enable streaming output from API
        Returns: previous value of stream
        """
        tmp = self.config.stream
        self.config.stream = stream
        return tmp

    def get_stream(self) -> bool:
        """Get streaming status"""
        return self.config.stream

    def _stream_response(  # type: ignore
        self, response, chat: bool = False
    ) -> Tuple[LLMResponse, OpenAIResponse]:
        """
        Grab and print streaming response from API.
        Args:
            response: event-sequence emitted by API
            chat: whether in chat-mode (or else completion-mode)
        Returns:
            Tuple consisting of:
                LLMResponse object (with message, usage),
                OpenAIResponse object (with choices, usage)

        """
        completion = ""
        function_args = ""
        function_name = ""

        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()
        has_function = False
        for event in response:
            event_args = ""
            event_fn_name = ""
            if chat:
                delta = event["choices"][0]["delta"]
                if "function_call" in delta:
                    if "name" in delta.function_call:
                        event_fn_name = delta.function_call["name"]
                    if "arguments" in delta.function_call:
                        event_args = delta.function_call["arguments"]
                event_text = delta.get("content", "")
            else:
                event_text = event["choices"][0]["text"]
            if event_text:
                completion += event_text
                sys.stdout.write(Colors().GREEN + event_text)
                sys.stdout.flush()
            if event_fn_name:
                function_name = event_fn_name
                has_function = True
                sys.stdout.write(Colors().GREEN + "FUNC: " + event_fn_name + ": ")
                sys.stdout.flush()
            if event_args:
                function_args += event_args
                sys.stdout.write(Colors().GREEN + event_args)
                sys.stdout.flush()
            if event.choices[0].finish_reason in ["stop", "function_call"]:
                # for function_call, finish_reason does not necessarily
                # contain "function_call" as mentioned in the docs.
                # So we check for "stop" or "function_call" here.
                break

        print("")
        # TODO- get usage info in stream mode (?)

        # check if function_call args are valid, if not,
        # treat this as a normal msg, not a function call
        args = {}
        if has_function and function_args != "":
            try:
                args = ast.literal_eval(function_args.strip())
            except (SyntaxError, ValueError):
                logging.warning(
                    f"Parsing OpenAI function args failed: {function_args};"
                    " treating args as normal message"
                )
                has_function = False
                completion = completion + function_args

        # mock openai response so we can cache it
        if chat:
            msg: Dict[str, Any] = dict(message=dict(content=completion))
            if has_function:
                function_call = LLMFunctionCall(name=function_name)
                function_call_dict = function_call.dict()
                if function_args == "":
                    function_call.arguments = None
                else:
                    function_call.arguments = args
                    function_call_dict.update({"arguments": function_args.strip()})
                msg["message"]["function_call"] = function_call_dict
        else:
            # non-chat mode has no function_call
            msg = dict(text=completion)

        openai_response = OpenAIResponse(
            choices=[msg],
            usage=dict(total_tokens=0),
        )
        return (  # type: ignore
            LLMResponse(
                message=completion,
                cached=False,
                function_call=function_call if has_function else None,
            ),
            openai_response.dict(),
        )

    def _cache_lookup(self, fn_name: str, **kwargs: Dict[str, Any]) -> Tuple[str, Any]:
        # Use the kwargs as the cache key
        sorted_kwargs_str = str(sorted(kwargs.items()))
        raw_key = f"{fn_name}:{sorted_kwargs_str}"

        # Hash the key to a fixed length using SHA256
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

        if not settings.cache:
            # when caching disabled, return the hashed_key and none result
            return hashed_key, None
        # Try to get the result from the cache
        return hashed_key, self.cache.retrieve(hashed_key)

    def _cost_chat_model(self, prompt: int, completion: int) -> float:
        price = self.chat_cost()
        return (price[0] * prompt + price[1] * completion) / 1000

    def _get_non_stream_token_usage(
        self, cached: bool, response: Dict[str, Any]
    ) -> LLMTokenUsage:
        """
        Extracts token usage from ``response`` and computes cost, only when NOT
        in streaming mode, since the LLM API (OpenAI currently) does not populate the
        usage fields in streaming mode. In streaming mode, these are set to zero for
        now, and will be updated later by the fn ``update_token_usage``.
        """
        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        if not cached and not self.config.stream:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            cost = self._cost_chat_model(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
        )

    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        try:
            return self._generate(prompt, max_tokens)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            err_msg = str(e)[:500]
            logging.error(f"OpenAI API error: {err_msg}")
            return LLMResponse(message=NO_ANSWER, cached=False)

    def _generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        if self.config.use_chat_for_completion:
            return self.chat(messages=prompt, max_tokens=max_tokens)
        openai.api_key = self.api_key
        if self.api_base:
            openai.api_base = self.api_base

        if settings.debug:
            print(f"[red]PROMPT: {prompt}[/red]")

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):  # type: ignore
            cached = False
            hashed_key, result = self._cache_lookup("Completion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    print("[red]CACHED[/red]")
            else:
                # If it's not in the cache, call the API
                result = openai.Completion.create(**kwargs)  # type: ignore
                if self.config.stream:
                    llm_response, openai_response = self._stream_response(result)
                    self.cache.store(hashed_key, openai_response)
                    return cached, hashed_key, openai_response
                else:
                    self.cache.store(hashed_key, result)
            return cached, hashed_key, result

        key_name = "engine" if self.config.type == "azure" else "model"
        cached, hashed_key, response = completions_with_backoff(
            **{key_name: self.config.completion_model},
            prompt=prompt,
            max_tokens=max_tokens,  # for output/completion
            request_timeout=self.config.timeout,
            temperature=self.config.temperature,
            echo=False,
            stream=self.config.stream,
        )

        msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, cached=cached)

    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        try:
            return await self._agenerate(prompt, max_tokens)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            err_msg = str(e)[:500]
            logging.error(f"OpenAI API error: {err_msg}")
            return LLMResponse(message=NO_ANSWER, cached=False)

    async def _agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key
        if self.api_base:
            openai.api_base = self.api_base
        # note we typically will not have self.config.stream = True
        # when issuing several api calls concurrently/asynchronously.
        # The calling fn should use the context `with Streaming(..., False)` to
        # disable streaming.
        if self.config.use_chat_for_completion:
            messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=prompt),
            ]

            @async_retry_with_exponential_backoff
            async def completions_with_backoff(
                **kwargs: Dict[str, Any]
            ) -> Tuple[bool, str, Any]:
                cached = False
                hashed_key, result = self._cache_lookup("AsyncChatCompletion", **kwargs)
                if result is not None:
                    cached = True
                else:
                    # If it's not in the cache, call the API
                    result = await openai.ChatCompletion.acreate(  # type: ignore
                        **kwargs
                    )
                    self.cache.store(hashed_key, result)
                return cached, hashed_key, result

            cached, hashed_key, response = await completions_with_backoff(
                model=self.config.chat_model,
                messages=[m.api_dict() for m in messages],
                max_tokens=max_tokens,
                request_timeout=self.config.timeout,
                temperature=self.config.temperature,
                stream=self.config.stream,
            )
            msg = response["choices"][0]["message"]["content"].strip()
        else:

            @retry_with_exponential_backoff
            async def completions_with_backoff(**kwargs):  # type: ignore
                cached = False
                hashed_key, result = self._cache_lookup("AsyncCompletion", **kwargs)
                if result is not None:
                    cached = True
                else:
                    # If it's not in the cache, call the API
                    result = await openai.Completion.acreate(**kwargs)  # type: ignore
                    self.cache.store(hashed_key, result)
                return cached, hashed_key, result

            cached, hashed_key, response = await completions_with_backoff(
                model=self.config.completion_model,
                prompt=prompt,
                max_tokens=max_tokens,
                request_timeout=self.config.timeout,
                temperature=self.config.temperature,
                echo=False,
                stream=self.config.stream,
            )
            msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, cached=cached)

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        if self.config.use_completion_for_chat:
            # only makes sense for local models
            if self.config.local is None or self.config.local.formatter is None:
                raise ValueError(
                    """
                    `formatter` must be specified in config to use completion for chat.
                    """
                )
            formatter = PromptFormatter.create(self.config.local.formatter)
            if isinstance(messages, str):
                messages = [
                    LLMMessage(
                        role=Role.SYSTEM, content="You are a helpful assistant."
                    ),
                    LLMMessage(role=Role.USER, content=messages),
                ]
            prompt = formatter.format(messages)
            return self.generate(prompt=prompt, max_tokens=max_tokens)
        try:
            return self._chat(messages, max_tokens, functions, function_call)
        except Exception as e:
            # capture exceptions not handled by retry, so we don't crash
            err_msg = str(e)[:500]
            logging.error(f"OpenAI API error: {err_msg}")
            return LLMResponse(message=NO_ANSWER, cached=False)

    def _chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        """
        ChatCompletion API call to OpenAI.
        Args:
            messages: list of messages  to send to the API, typically
                represents back and forth dialogue between user and LLM, but could
                also include "function"-role messages. If messages is a string,
                it is assumed to be a user message.
            max_tokens: max output tokens to generate
            functions: list of LLMFunction specs available to the LLM, to possibly
                use in its response
            function_call: controls how the LLM uses `functions`:
                - "auto": LLM decides whether to use `functions` or not,
                - "none": LLM blocked from using any function
                - a dict of {"name": "function_name"} which forces the LLM to use
                    the specified function.
        Returns:
            LLMResponse object
        """
        openai.api_key = self.api_key
        if self.api_base:
            openai.api_base = self.api_base
        if isinstance(messages, str):
            llm_messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages),
            ]
        else:
            llm_messages = messages

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):  # type: ignore
            cached = False
            hashed_key, result = self._cache_lookup("Completion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    print("[red]CACHED[/red]")
            else:
                # If it's not in the cache, call the API
                result = openai.ChatCompletion.create(**kwargs)  # type: ignore
                if not self.config.stream:
                    # if streaming, cannot cache result
                    # since it is a generator. Instead,
                    # we hold on to the hashed_key and
                    # cache the result later
                    self.cache.store(hashed_key, result)
            return cached, hashed_key, result

        # Azure uses different parameters. It uses ``engine`` instead of ``model``
        # and the value should be the deployment_name not ``self.config.chat_model``
        chat_model = self.config.chat_model
        key_name = "model"
        if self.config.type == "azure":
            key_name = "engine"
            if hasattr(self, "deployment_name"):
                chat_model = self.deployment_name

        args: Dict[str, Any] = dict(
            **{key_name: chat_model},
            messages=[m.api_dict() for m in llm_messages],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=self.config.temperature,
            request_timeout=self.config.timeout,
            stream=self.config.stream,
        )
        # only include functions-related args if functions are provided
        # since the OpenAI API will throw an error if `functions` is None or []
        if functions is not None:
            args.update(
                dict(
                    functions=[f.dict() for f in functions],
                    function_call=function_call,
                )
            )
        cached, hashed_key, response = completions_with_backoff(**args)

        if self.config.stream and not cached:
            llm_response, openai_response = self._stream_response(response, chat=True)
            self.cache.store(hashed_key, openai_response)
            return llm_response

        # openAI response will look like this:
        """
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "name": "", 
                    "content": "\n\nHello there, how may I help you?",
                    "function_call": {
                        "name": "fun_name",
                        "arguments: {
                            "arg1": "val1",
                            "arg2": "val2"
                        }
                    }, 
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        """
        message = response["choices"][0]["message"]
        msg = message["content"] or ""
        if message.get("function_call") is None:
            fun_call = None
        else:
            fun_call = LLMFunctionCall(name=message["function_call"]["name"])
            try:
                fun_args = ast.literal_eval(message["function_call"]["arguments"])
                fun_call.arguments = fun_args
            except (ValueError, SyntaxError):
                logging.warning(
                    "Could not parse function arguments: "
                    f"{message['function_call']['arguments']} "
                    f"for function {message['function_call']['name']} "
                    "treating as normal non-function message"
                )
                fun_call = None
                msg = message["content"] + message["function_call"]["arguments"]

        return LLMResponse(
            message=msg.strip() if msg is not None else "",
            function_call=fun_call,
            cached=cached,
            usage=self._get_non_stream_token_usage(cached, response),
        )
