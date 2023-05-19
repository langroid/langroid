from llmagent.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMResponse,
    LLMMessage,
)
import sys
from llmagent.language_models.utils import retry_with_exponential_backoff
from llmagent.utils.configuration import settings
from llmagent.utils.constants import Colors
from llmagent.utils.output.printing import PrintColored
from llmagent.cachedb.redis_cachedb import RedisCache
from pydantic import BaseModel
import hashlib
from typing import List, Tuple, Dict
import openai
from dotenv import load_dotenv
import os
import logging

logging.getLogger("openai").setLevel(logging.ERROR)


class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    max_tokens: int = 1024
    chat_model: str = "gpt-3.5-turbo"
    completion_model: str = "text-davinci-003"


class OpenAIResponse(BaseModel):
    """OpenAI response model, either completion or chat."""

    choices: List[Dict]
    usage: Dict


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
        super().__init__()
        self.config = config
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
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

    def _stream_response(
        self, response, chat=False
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
        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()
        for event in response:
            if chat:
                event_text = event["choices"][0]["delta"].get("content", "")
            else:
                event_text = event["choices"][0]["text"]
            if event_text:
                completion += event_text
                sys.stdout.write(Colors().GREEN + event_text)
                sys.stdout.flush()

        print(Colors().RESET)
        # TODO- get usage info in stream mode (?)
        if settings.debug:
            with PrintColored(Colors().RED):
                print(Colors().RED + f"LLM: {completion}")

        # mock openai response so we can cache it
        if chat:
            msg = dict(message=dict(content=completion))
        else:
            msg = dict(text=completion)
        openai_response = OpenAIResponse(
            choices=[msg],
            usage=dict(total_tokens=0),
        )
        return LLMResponse(message=completion, usage=0), openai_response.dict()

    def _cache_lookup(self, fn_name: str, **kwargs):
        # Use the kwargs as the cache key
        sorted_kwargs_str = str(sorted(kwargs.items()))
        raw_key = f"{fn_name}:{sorted_kwargs_str}"

        # Hash the key to a fixed length using SHA256
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

        if not settings.cache:
            return hashed_key, None
        # Try to get the result from the cache
        return hashed_key, self.cache.retrieve(hashed_key)

    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key

        if settings.debug:
            with PrintColored(Colors().RED):
                print(Colors().RED + f"PROMPT: {prompt}")

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):
            cached = False
            hashed_key, result = self._cache_lookup("Completion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    with PrintColored(Colors().RED):
                        print(Colors().RED + "CACHED")
            else:
                # If it's not in the cache, call the API
                result = openai.Completion.create(**kwargs)
                if not self.config.stream:
                    # if streaming, cannot cache result
                    # since it is a generator. Instead,
                    # we hold on to the hashed_key and
                    # cache the result later
                    self.cache.store(hashed_key, result)
            return cached, hashed_key, result

        cached, hashed_key, response = completions_with_backoff(
            model=self.config.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
            stream=self.config.stream,
        )
        if self.config.stream and not cached:
            llm_response, openai_response = self._stream_response(response)
            self.cache.store(hashed_key, openai_response)
            return llm_response
        usage = response["usage"]["total_tokens"]
        msg = response["choices"][0]["text"].strip()
        if settings.debug:
            with PrintColored(Colors().RED):
                print(Colors().RED + f"LLM: {msg}")
        return LLMResponse(message=msg, usage=usage, cached=cached)

    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        # TODO: implement caching, streaming, retry for async
        openai.api_key = self.api_key
        # note we typically will not have self.config.stream = True
        # when issuing several api calls concurrently/asynchronously.
        # The calling fn should use the context `with Streaming(..., False)` to
        # disable streaming.
        response = await openai.Completion.acreate(
            model=self.config.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
            stream=self.config.stream,
        )
        usage = response["usage"]["total_tokens"]
        msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, usage=usage)

    def chat(self, messages: List[LLMMessage], max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):
            cached = False
            hashed_key, result = self._cache_lookup("Completion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    with PrintColored(Colors().RED):
                        print(Colors().RED + "CACHED")
            else:
                # If it's not in the cache, call the API
                result = openai.ChatCompletion.create(**kwargs)
                if not self.config.stream:
                    # if streaming, cannot cache result
                    # since it is a generator. Instead,
                    # we hold on to the hashed_key and
                    # cache the result later
                    self.cache.store(hashed_key, result)
            return cached, hashed_key, result

        cached, hashed_key, response = completions_with_backoff(
            model=self.config.chat_model,
            messages=[m.dict() for m in messages],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
            stream=self.config.stream,
        )
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
                    "content": "\n\nHello there, how may I help you?",
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

        msg = response["choices"][0]["message"]["content"].strip()
        return LLMResponse(message=msg, usage=usage, cached=cached)
