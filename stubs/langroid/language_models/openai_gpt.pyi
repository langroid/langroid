from enum import Enum
from typing import Any, Callable

from _typeshed import Incomplete
from groq import AsyncGroq, Groq
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from langroid.cachedb.base import CacheDB as CacheDB
from langroid.cachedb.redis_cachedb import (
    RedisCache as RedisCache,
)
from langroid.cachedb.redis_cachedb import (
    RedisCacheConfig as RedisCacheConfig,
)
from langroid.exceptions import LangroidImportError as LangroidImportError
from langroid.language_models.base import (
    LanguageModel as LanguageModel,
)
from langroid.language_models.base import (
    LLMConfig as LLMConfig,
)
from langroid.language_models.base import (
    LLMFunctionCall as LLMFunctionCall,
)
from langroid.language_models.base import (
    LLMFunctionSpec as LLMFunctionSpec,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.base import (
    LLMResponse as LLMResponse,
)
from langroid.language_models.base import (
    LLMTokenUsage as LLMTokenUsage,
)
from langroid.language_models.base import (
    Role as Role,
)
from langroid.language_models.config import (
    HFPromptFormatterConfig as HFPromptFormatterConfig,
)
from langroid.language_models.prompt_formatter.hf_formatter import (
    HFFormatter as HFFormatter,
)
from langroid.language_models.prompt_formatter.hf_formatter import (
    find_hf_formatter as find_hf_formatter,
)
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff as async_retry_with_exponential_backoff,
)
from langroid.language_models.utils import (
    retry_with_exponential_backoff as retry_with_exponential_backoff,
)
from langroid.utils.configuration import settings as settings
from langroid.utils.constants import Colors as Colors
from langroid.utils.system import friendly_error as friendly_error

OLLAMA_BASE_URL: Incomplete
OLLAMA_API_KEY: str
DUMMY_API_KEY: str

class OpenAIChatModel(str, Enum):
    GPT3_5_TURBO: str
    GPT4: str
    GPT4_32K: str
    GPT4_TURBO: str
    GPT4o: str

class OpenAICompletionModel(str, Enum):
    TEXT_DA_VINCI_003: str
    GPT3_5_TURBO_INSTRUCT: str

openAIChatModelPreferenceList: Incomplete
openAICompletionModelPreferenceList: Incomplete
available_models: Incomplete
defaultOpenAIChatModel: Incomplete
defaultOpenAICompletionModel: Incomplete

class AccessWarning(Warning): ...

def gpt_3_5_warning() -> None: ...
def noop() -> None: ...

class OpenAICallParams(BaseModel):
    max_tokens: int
    temperature: float
    frequency_penalty: float | None
    presence_penalty: float | None
    response_format: dict[str, str] | None
    logit_bias: dict[int, float] | None
    logprobs: bool
    top_p: int | None
    top_logprobs: int | None
    n: int
    stop: str | list[str] | None
    seed: int | None
    user: str | None
    def to_dict_exclude_none(self) -> dict[str, Any]: ...

class OpenAIGPTConfig(LLMConfig):
    type: str
    api_key: str
    organization: str
    api_base: str | None
    litellm: bool
    ollama: bool
    max_output_tokens: int
    min_output_tokens: int
    use_chat_for_completion: bool
    timeout: int
    temperature: float
    seed: int | None
    params: OpenAICallParams | None
    chat_model: str
    completion_model: str
    run_on_first_use: Callable[[], None]
    formatter: str | None
    hf_formatter: HFFormatter | None
    def __init__(self, **kwargs) -> None: ...

    class Config:
        env_prefix: str

    @classmethod
    def create(cls, prefix: str) -> type["OpenAIGPTConfig"]: ...

class OpenAIResponse(BaseModel):
    choices: list[dict]
    usage: dict

def litellm_logging_fn(model_call_dict: dict[str, Any]) -> None: ...

class OpenAIGPT(LanguageModel):
    client: OpenAI | Groq
    async_client: AsyncOpenAI | AsyncGroq
    config: Incomplete
    run_on_first_use: Incomplete
    api_base: Incomplete
    api_key: Incomplete
    is_groq: Incomplete
    cache: Incomplete
    def __init__(self, config: OpenAIGPTConfig = ...) -> None: ...
    def is_openai_chat_model(self) -> bool: ...
    def is_openai_completion_model(self) -> bool: ...
    def chat_context_length(self) -> int: ...
    def completion_context_length(self) -> int: ...
    def chat_cost(self) -> tuple[float, float]: ...
    def set_stream(self, stream: bool) -> bool: ...
    def get_stream(self) -> bool: ...
    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse: ...
    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse: ...
    def chat(
        self,
        messages: str | list[LLMMessage],
        max_tokens: int = 200,
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "auto",
    ) -> LLMResponse: ...
    async def achat(
        self,
        messages: str | list[LLMMessage],
        max_tokens: int = 200,
        functions: list[LLMFunctionSpec] | None = None,
        function_call: str | dict[str, str] = "auto",
    ) -> LLMResponse: ...
