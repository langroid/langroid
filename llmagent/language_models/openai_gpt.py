from llmagent.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMResponse,
    LLMMessage,
)
import sys
from llmagent.utils.llms.rate_limits import retry_with_exponential_backoff
from llmagent.utils.constants import Colors
from typing import List
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

    def _stream_response(self, response, chat=False):
        """
        Grab and print streaming response from API.
        Args:
            response: event-sequence emitted by API
            chat: whether in chat-mode (or else completion-mode)
        Returns:
            LLMResponse object (with message, usage)

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
        return LLMResponse(message=completion, usage=0)

    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        openai.api_key = self.api_key

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):
            return openai.Completion.create(**kwargs)

        response = completions_with_backoff(
            model=self.config.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
            stream=self.config.stream,
        )
        if self.config.stream:
            return self._stream_response(response)
        else:
            usage = response["usage"]["total_tokens"]
            msg = response["choices"][0]["text"].strip()
            return LLMResponse(message=msg, usage=usage)

    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
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
            return openai.ChatCompletion.create(**kwargs)

        response = completions_with_backoff(
            model=self.config.chat_model,
            messages=[m.dict() for m in messages],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
            stream=self.config.stream,
        )
        if self.config.stream:
            return self._stream_response(response, chat=True)

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
        return LLMResponse(message=msg, usage=usage)
