import abc
from abc import ABC, abstractmethod

from _typeshed import Incomplete

from langroid.language_models.base import LLMMessage as LLMMessage
from langroid.language_models.config import (
    PromptFormatterConfig as PromptFormatterConfig,
)

logger: Incomplete

class PromptFormatter(ABC, metaclass=abc.ABCMeta):
    config: Incomplete
    def __init__(self, config: PromptFormatterConfig) -> None: ...
    @staticmethod
    def create(formatter: str) -> PromptFormatter: ...
    @abstractmethod
    def format(self, messages: list[LLMMessage]) -> str: ...
