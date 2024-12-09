"""Mock Language Model for testing"""

from typing import Awaitable, Callable, Dict, List, Optional, Union

import langroid.language_models as lm
from langroid.language_models import LLMResponse
from langroid.language_models.base import (
    LanguageModel,
    LLMConfig,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
    ToolChoiceTypes,
)
from langroid.utils.types import to_string


def none_fn(x: str) -> None | str:
    return None


class MockLMConfig(LLMConfig):
    """
    Mock Language Model Configuration.

    Attributes:
        response_dict (Dict[str, str]): A "response rule-book", in the form of a
            dictionary; if last msg in dialog is x,then respond with response_dict[x]
    """

    response_dict: Dict[str, str] = {}
    response_fn: Callable[[str], None | str] = none_fn
    response_fn_async: Optional[Callable[[str], Awaitable[Optional[str]]]] = None
    default_response: str = "Mock response"

    type: str = "mock"


class MockLM(LanguageModel):

    def __init__(self, config: MockLMConfig = MockLMConfig()):
        super().__init__(config)
        self.config: MockLMConfig = config

    def _response(self, msg: str) -> LLMResponse:
        # response is based on this fallback order:
        # - response_dict
        # - response_fn
        # - default_response
        mapped_response = self.config.response_dict.get(
            msg, self.config.response_fn(msg) or self.config.default_response
        )
        return lm.LLMResponse(
            message=to_string(mapped_response),
            cached=False,
        )

    async def _response_async(self, msg: str) -> LLMResponse:
        # response is based on this fallback order:
        # - response_dict
        # - response_fn_async
        # - response_fn
        # - default_response
        if self.config.response_fn_async is not None:
            response = await self.config.response_fn_async(msg)
        else:
            response = self.config.response_fn(msg)

        mapped_response = self.config.response_dict.get(
            msg, response or self.config.default_response
        )
        return lm.LLMResponse(
            message=to_string(mapped_response),
            cached=False,
        )

    def chat(
        self,
        messages: Union[str, List[lm.LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[lm.LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> lm.LLMResponse:
        """
        Mock chat function for testing
        """
        last_msg = messages[-1].content if isinstance(messages, list) else messages
        return self._response(last_msg)

    async def achat(
        self,
        messages: Union[str, List[lm.LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[lm.LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> lm.LLMResponse:
        """
        Mock chat function for testing
        """
        last_msg = messages[-1].content if isinstance(messages, list) else messages
        return await self._response_async(last_msg)

    def generate(self, prompt: str, max_tokens: int = 200) -> lm.LLMResponse:
        """
        Mock generate function for testing
        """
        return self._response(prompt)

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        """
        Mock generate function for testing
        """
        return await self._response_async(prompt)

    def get_stream(self) -> bool:
        return False

    def set_stream(self, stream: bool) -> bool:
        return False
