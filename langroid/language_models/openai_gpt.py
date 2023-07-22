import ast
import hashlib
import logging
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import openai
from dotenv import load_dotenv
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
    Role,
)
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)
from langroid.utils.configuration import settings
from langroid.utils.constants import Colors

logging.getLogger("openai").setLevel(logging.ERROR)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT3_5_TURBO = "gpt-3.5-turbo-0613"
    GPT4_NOFUNC = "gpt-4"  # before function_call API
    GPT4 = "gpt-4-0613"


class OpenAICompletionModel(str, Enum):
    """Enum for OpenAI Completion models"""

    TEXT_DA_VINCI_003 = "text-davinci-003"
    TEXT_ADA_001 = "text-ada-001"
    GPT4 = "gpt-4-0613"


class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    max_output_tokens: int = 1024
    min_output_tokens: int = 64
    timeout: int = 20
    temperature: float = 0.0
    chat_model: OpenAIChatModel = OpenAIChatModel.GPT4
    completion_model: OpenAICompletionModel = OpenAICompletionModel.GPT4
    context_length: Dict[str, int] = {
        OpenAIChatModel.GPT3_5_TURBO: 4096,
        OpenAIChatModel.GPT4: 8192,
        OpenAIChatModel.GPT4_NOFUNC: 8192,
        OpenAICompletionModel.TEXT_DA_VINCI_003: 4096,
    }


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
        load_dotenv()
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
                usage=0,
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
            # when cacheing disabled, return the hashed_key and none result
            return hashed_key, None
        # Try to get the result from the cache
        return hashed_key, self.cache.retrieve(hashed_key)

    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        if self.config.use_chat_for_completion:
            return self.chat(messages=prompt, max_tokens=max_tokens)
        openai.api_key = self.api_key

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

        cached, hashed_key, response = completions_with_backoff(
            model=self.config.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,  # for output/completion
            request_timeout=self.config.timeout,
            temperature=self.config.temperature,
            echo=False,
            stream=self.config.stream,
        )

        usage = response["usage"]["total_tokens"]
        msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, usage=usage, cached=cached)

    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key
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
                temperature=0,
                stream=self.config.stream,
            )
            usage = response["usage"]["total_tokens"]
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
                temperature=0,
                echo=False,
                stream=self.config.stream,
            )
            usage = response["usage"]["total_tokens"]
            msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, usage=usage, cached=cached)

    def chat(
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
        if type(messages) == str:
            llm_messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages),
            ]
        else:
            llm_messages = cast(List[LLMMessage], messages)

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

        args: Dict[str, Any] = dict(
            model=self.config.chat_model,
            messages=[m.api_dict() for m in llm_messages],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
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

        usage = response["usage"]["total_tokens"]
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
                    f"Could not parse function arguments: "
                    f"{message['function_call']['arguments']} "
                    f"for function {message['function_call']['name']} "
                    f"treating as normal non-function message"
                )
                fun_call = None
                msg = message["content"] + message["function_call"]["arguments"]

        return LLMResponse(
            message=msg.strip() if msg is not None else "",
            function_call=fun_call,
            usage=usage,
            cached=cached,
        )
