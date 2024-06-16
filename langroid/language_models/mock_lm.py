"""Mock Language Model for testing"""

from typing import Callable, Dict, List, Optional, Union

import langroid.language_models as lm
from langroid.language_models import LLMResponse
from langroid.language_models.base import LanguageModel, LLMConfig


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
        return lm.LLMResponse(
            message=self.config.response_dict.get(
                msg,
                self.config.response_fn(msg) or self.config.default_response,
            ),
            cached=False,
        )

    def chat(
        self,
        messages: Union[str, List[lm.LLMMessage]],
        max_tokens: int = 200,
        functions: Optional[List[lm.LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
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
        functions: Optional[List[lm.LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> lm.LLMResponse:
        """
        Mock chat function for testing
        """
        last_msg = messages[-1].content if isinstance(messages, list) else messages
        return self._response(last_msg)

    def generate(self, prompt: str, max_tokens: int = 200) -> lm.LLMResponse:
        """
        Mock generate function for testing
        """
        return self._response(prompt)

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        """
        Mock generate function for testing
        """
        return self._response(prompt)

    def get_stream(self) -> bool:
        return False

    def set_stream(self, stream: bool) -> bool:
        return False
