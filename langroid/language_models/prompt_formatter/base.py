import logging
from abc import ABC, abstractmethod
from typing import List

from langroid.language_models.base import LLMMessage
from langroid.language_models.config import PromptFormatterConfig

logger = logging.getLogger(__name__)


class PromptFormatter(ABC):
    """
    Abstract base class for a prompt formatter
    """

    def __init__(self, config: PromptFormatterConfig):
        self.config = config

    @staticmethod
    def create(config: PromptFormatterConfig) -> "PromptFormatter":
        from langroid.language_models.prompt_formatter.llama2_formatter import (
            Llama2Formatter,
        )

        formatter_class = dict(llama2=Llama2Formatter).get(config.type, Llama2Formatter)
        return formatter_class(config)

    @abstractmethod
    def format(self, messages: List[LLMMessage]) -> str:
        """
        Convert sequence of messages (system, user, assistant, user, assistant...user)
            to a single prompt formatted according to the specific format type,
            to be used in a /completions endpoint.

        Args:
            messages (List[LLMMessage]): chat history as a sequence of messages

        Returns:
            (str): formatted version of chat history

        """
        pass
