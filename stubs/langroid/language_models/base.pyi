import abc
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from _typeshed import Incomplete
from pydantic import BaseModel, BaseSettings

from langroid.cachedb.base import CacheDBConfig as CacheDBConfig
from langroid.mytypes import Document as Document
from langroid.parsing.agent_chats import parse_message as parse_message
from langroid.parsing.parse_json import top_level_json_field as top_level_json_field
from langroid.prompts.dialog import collate_chat_history as collate_chat_history
from langroid.prompts.templates import (
    EXTRACTION_PROMPT_GPT4 as EXTRACTION_PROMPT_GPT4,
)
from langroid.prompts.templates import (
    SUMMARY_ANSWER_PROMPT_GPT4 as SUMMARY_ANSWER_PROMPT_GPT4,
)
from langroid.utils.configuration import settings as settings
from langroid.utils.output.printing import show_if_debug as show_if_debug

logger: Incomplete

def noop_fn(*args: list[Any], **kwargs: dict[str, Any]) -> None: ...

class LLMConfig(BaseSettings):
    type: str
    streamer: Callable[[Any], None] | None
    api_base: str | None
    formatter: None | str
    timeout: int
    chat_model: str
    completion_model: str
    temperature: float
    chat_context_length: int
    completion_context_length: int
    max_output_tokens: int
    min_output_tokens: int
    use_completion_for_chat: bool
    use_chat_for_completion: bool
    stream: bool
    cache_config: None | CacheDBConfig
    chat_cost_per_1k_tokens: tuple[float, float]
    completion_cost_per_1k_tokens: tuple[float, float]

class LLMFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] | None
    @staticmethod
    def from_dict(message: dict[str, Any]) -> LLMFunctionCall: ...

class LLMFunctionSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

class LLMTokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost: float
    calls: int
    def reset(self) -> None: ...
    @property
    def total_tokens(self) -> int: ...

class Role(str, Enum):
    USER: str
    SYSTEM: str
    ASSISTANT: str
    FUNCTION: str

class LLMMessage(BaseModel):
    role: Role
    name: str | None
    tool_id: str
    content: str
    function_call: LLMFunctionCall | None
    timestamp: datetime
    def api_dict(self) -> dict[str, Any]: ...

class LLMResponse(BaseModel):
    message: str
    tool_id: str
    function_call: LLMFunctionCall | None
    usage: LLMTokenUsage | None
    cached: bool
    def to_LLMMessage(self) -> LLMMessage: ...
    def get_recipient_and_message(self) -> tuple[str, str]: ...

class LanguageModel(ABC, metaclass=abc.ABCMeta):
    usage_cost_dict: dict[str, LLMTokenUsage]
    config: Incomplete
    def __init__(self, config: LLMConfig = ...) -> None: ...
    @staticmethod
    def create(config: LLMConfig | None) -> LanguageModel | None: ...
    @staticmethod
    def user_assistant_pairs(lst: list[str]) -> list[tuple[str, str]]: ...
    @staticmethod
    def get_chat_history_components(
        messages: list[LLMMessage],
    ) -> tuple[str, list[tuple[str, str]], str]: ...
    @abstractmethod
    def set_stream(self, stream: bool) -> bool: ...
    @abstractmethod
    def get_stream(self) -> bool: ...
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse: ...
    @abstractmethod
    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse: ...
    @abstractmethod
    def chat(
        self,
        messages: str | list[LLMMessage],
        max_tokens: int = 200,
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "auto",
    ) -> LLMResponse: ...
    @abstractmethod
    async def achat(
        self,
        messages: str | list[LLMMessage],
        max_tokens: int = 200,
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "auto",
    ) -> LLMResponse: ...
    def __call__(self, prompt: str, max_tokens: int) -> LLMResponse: ...
    def chat_context_length(self) -> int: ...
    def completion_context_length(self) -> int: ...
    def chat_cost(self) -> tuple[float, float]: ...
    def reset_usage_cost(self) -> None: ...
    def update_usage_cost(
        self, chat: bool, prompts: int, completions: int, cost: float
    ) -> None: ...
    @classmethod
    def usage_cost_summary(cls) -> str: ...
    @classmethod
    def tot_tokens_cost(cls) -> tuple[int, float]: ...
    def followup_to_standalone(
        self, chat_history: list[tuple[str, str]], question: str
    ) -> str: ...
    async def get_verbatim_extract_async(
        self, question: str, passage: Document
    ) -> str: ...
    def get_verbatim_extracts(
        self, question: str, passages: list[Document]
    ) -> list[Document]: ...
    def get_summary_answer(
        self, question: str, passages: list[Document]
    ) -> Document: ...

class StreamingIfAllowed:
    llm: Incomplete
    stream: Incomplete
    def __init__(self, llm: LanguageModel, stream: bool = True) -> None: ...
    old_stream: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
