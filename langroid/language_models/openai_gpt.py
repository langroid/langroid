import hashlib
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from functools import cache
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import openai
from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from groq import AsyncGroq, Groq
from httpx import Timeout
from openai import AsyncOpenAI, OpenAI
from rich import print
from rich.markup import escape

from langroid.cachedb.base import CacheDB
from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.exceptions import LangroidImportError
from langroid.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMFunctionCall,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    LLMTokenUsage,
    OpenAIJsonSchemaSpec,
    OpenAIToolCall,
    OpenAIToolSpec,
    Role,
    StreamEventType,
    ToolChoiceTypes,
)
from langroid.language_models.config import HFPromptFormatterConfig
from langroid.language_models.model_info import (
    DeepSeekModel,
    OpenAI_API_ParamInfo,
)
from langroid.language_models.model_info import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.model_info import (
    OpenAICompletionModel as OpenAICompletionModel,
)
from langroid.language_models.prompt_formatter.hf_formatter import (
    HFFormatter,
    find_hf_formatter,
)
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)
from langroid.parsing.parse_json import parse_imperfect_json
from langroid.pydantic_v1 import BaseModel, BaseSettings
from langroid.utils.configuration import settings
from langroid.utils.constants import Colors
from langroid.utils.system import friendly_error

logging.getLogger("openai").setLevel(logging.ERROR)

if "OLLAMA_HOST" in os.environ:
    OLLAMA_BASE_URL = f"http://{os.environ['OLLAMA_HOST']}/v1"
else:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GLHF_BASE_URL = "https://glhf.chat/api/openai/v1"
LANGDB_BASE_URL = "https://api.us-east-1.langdb.ai"
OLLAMA_API_KEY = "ollama"
DUMMY_API_KEY = "xxx"

VLLM_API_KEY = os.environ.get("VLLM_API_KEY", DUMMY_API_KEY)
LLAMACPP_API_KEY = os.environ.get("LLAMA_API_KEY", DUMMY_API_KEY)


openai_chat_model_pref_list = [
    OpenAIChatModel.GPT4o,
    OpenAIChatModel.GPT4o_MINI,
    OpenAIChatModel.O1_MINI,
    OpenAIChatModel.O1,
    OpenAIChatModel.GPT3_5_TURBO,
]

openai_completion_model_pref_list = [
    OpenAICompletionModel.DAVINCI,
    OpenAICompletionModel.BABBAGE,
]


if "OPENAI_API_KEY" in os.environ:
    try:
        available_models = set(map(lambda m: m.id, OpenAI().models.list()))
    except openai.AuthenticationError as e:
        if settings.debug:
            logging.warning(
                f"""
            OpenAI Authentication Error: {e}.
            ---
            If you intended to use an OpenAI Model, you should fix this,
            otherwise you can ignore this warning.
            """
            )
        available_models = set()
    except Exception as e:
        if settings.debug:
            logging.warning(
                f"""
            Error while fetching available OpenAI models: {e}.
            Proceeding with an empty set of available models.
            """
            )
        available_models = set()
else:
    available_models = set()

default_openai_chat_model = next(
    chain(
        filter(
            lambda m: m.value in available_models,
            openai_chat_model_pref_list,
        ),
        [OpenAIChatModel.GPT4o],
    )
)
default_openai_completion_model = next(
    chain(
        filter(
            lambda m: m.value in available_models,
            openai_completion_model_pref_list,
        ),
        [OpenAICompletionModel.DAVINCI],
    )
)


class AccessWarning(Warning):
    pass


@cache
def gpt_3_5_warning() -> None:
    warnings.warn(
        f"""
        {OpenAIChatModel.GPT4o} is not available,
        falling back to {OpenAIChatModel.GPT3_5_TURBO}.
        Examples may not work properly and unexpected behavior may occur.
        Adjustments to prompts may be necessary.
        """,
        AccessWarning,
    )


@cache
def parallel_strict_warning() -> None:
    logging.warning(
        "OpenAI tool calling in strict mode is not supported when "
        "parallel tool calls are made. Disable parallel tool calling "
        "to ensure correct behavior."
    )


def noop() -> None:
    """Does nothing."""
    return None


class LangDBParams(BaseSettings):
    """
    Parameters specific to LangDB integration.
    """

    api_key: str = DUMMY_API_KEY
    project_id: str = ""
    label: Optional[str] = None
    run_id: Optional[str] = None
    thread_id: Optional[str] = None
    base_url: str = LANGDB_BASE_URL

    class Config:
        # allow setting of fields via env vars,
        # e.g. LANGDB_PROJECT_ID=1234
        env_prefix = "LANGDB_"


class OpenAICallParams(BaseModel):
    """
    Various params that can be sent to an OpenAI API chat-completion call.
    When specified, any param here overrides the one with same name in the
    OpenAIGPTConfig.
    See OpenAI API Reference for details on the params:
    https://platform.openai.com/docs/api-reference/chat
    """

    max_tokens: int = 1024
    temperature: float = 0.2
    frequency_penalty: float | None = None  # between -2 and 2
    presence_penalty: float | None = None  # between -2 and 2
    response_format: Dict[str, str] | None = None
    logit_bias: Dict[int, float] | None = None  # token_id -> bias
    logprobs: bool | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None  # or "low" or "high" or "medium"
    top_logprobs: int | None = None  # if int, requires logprobs=True
    n: int = 1  # how many completions to generate (n > 1 is NOT handled now)
    stop: str | List[str] | None = None  # (list of) stop sequence(s)
    seed: int | None = None
    user: str | None = None  # user id for tracking
    extra_body: Dict[str, Any] | None = None  # additional params for API request body

    def to_dict_exclude_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.dict().items() if v is not None}


class LiteLLMProxyConfig(BaseSettings):
    """Configuration for LiteLLM proxy connection."""

    api_key: str = ""  # read from env var LITELLM_API_KEY if set
    api_base: str = ""  # read from env var LITELLM_API_BASE if set

    class Config:
        env_prefix = "LITELLM_"


class OpenAIGPTConfig(LLMConfig):
    """
    Class for any LLM with an OpenAI-like API: besides the OpenAI models this includes:
    (a) locally-served models behind an OpenAI-compatible API
    (b) non-local models, using a proxy adaptor lib like litellm that provides
        an OpenAI-compatible API.
    (We could rename this class to OpenAILikeConfig, but we keep it as-is for now)

    Important Note:
    Due to the `env_prefix = "OPENAI_"` defined below,
    all of the fields below can be set AND OVERRIDDEN via env vars,
    # by upper-casing the name and prefixing with OPENAI_, e.g.
    # OPENAI_MAX_OUTPUT_TOKENS=1000.
    # If any of these is defined in this way in the environment
    # (either via explicit setenv or export or via .env file + load_dotenv()),
    # the environment variable takes precedence over the value in the config.
    """

    type: str = "openai"
    api_key: str = DUMMY_API_KEY
    organization: str = ""
    api_base: str | None = None  # used for local or other non-OpenAI models
    litellm: bool = False  # use litellm api?
    litellm_proxy: LiteLLMProxyConfig = LiteLLMProxyConfig()
    ollama: bool = False  # use ollama's OpenAI-compatible endpoint?
    min_output_tokens: int = 1
    use_chat_for_completion = True  # do not change this, for OpenAI models!
    timeout: int = 20
    temperature: float = 0.2
    seed: int | None = 42
    params: OpenAICallParams | None = None
    # these can be any model name that is served at an OpenAI-compatible API end point
    chat_model: str = default_openai_chat_model
    chat_model_orig: str = default_openai_chat_model
    completion_model: str = default_openai_completion_model
    run_on_first_use: Callable[[], None] = noop
    parallel_tool_calls: Optional[bool] = None
    # Supports constrained decoding which enforces that the output of the LLM
    # adheres to a JSON schema
    supports_json_schema: Optional[bool] = None
    # Supports strict decoding for the generation of tool calls with
    # the OpenAI Tools API; this ensures that the generated tools
    # adhere to the provided schema.
    supports_strict_tools: Optional[bool] = None
    # a string that roughly matches a HuggingFace chat_template,
    # e.g. "mistral-instruct-v0.2 (a fuzzy search is done to find the closest match)
    formatter: str | None = None
    hf_formatter: HFFormatter | None = None
    langdb_params: LangDBParams = LangDBParams()
    headers: Dict[str, str] = {}

    def __init__(self, **kwargs) -> None:  # type: ignore
        local_model = "api_base" in kwargs and kwargs["api_base"] is not None

        chat_model = kwargs.get("chat_model", "")
        local_prefixes = ["local/", "litellm/", "ollama/", "vllm/", "llamacpp/"]
        if any(chat_model.startswith(prefix) for prefix in local_prefixes):
            local_model = True

        warn_gpt_3_5 = (
            "chat_model" not in kwargs.keys()
            and not local_model
            and default_openai_chat_model == OpenAIChatModel.GPT3_5_TURBO
        )

        if warn_gpt_3_5:
            existing_hook = kwargs.get("run_on_first_use", noop)

            def with_warning() -> None:
                existing_hook()
                gpt_3_5_warning()

            kwargs["run_on_first_use"] = with_warning

        super().__init__(**kwargs)

    class Config:
        env_prefix = "OPENAI_"

    def _validate_litellm(self) -> None:
        """
        When using liteLLM, validate whether all env vars required by the model
        have been set.
        """
        if not self.litellm:
            return
        try:
            import litellm
        except ImportError:
            raise LangroidImportError("litellm", "litellm")
        litellm.telemetry = False
        litellm.drop_params = True  # drop un-supported params without crashing
        # modify params to fit the model expectations, and avoid crashing
        # (e.g. anthropic doesn't like first msg to be system msg)
        litellm.modify_params = True
        self.seed = None  # some local mdls don't support seed
        keys_dict = litellm.utils.validate_environment(self.chat_model)
        missing_keys = keys_dict.get("missing_keys", [])
        if len(missing_keys) > 0:
            raise ValueError(
                f"""
                Missing environment variables for litellm-proxied model:
                {missing_keys}
                """
            )

    @classmethod
    def create(cls, prefix: str) -> Type["OpenAIGPTConfig"]:
        """Create a config class whose params can be set via a desired
        prefix from the .env file or env vars.
        E.g., using
        ```python
        OllamaConfig = OpenAIGPTConfig.create("ollama")
        ollama_config = OllamaConfig()
        ```
        you can have a group of params prefixed by "OLLAMA_", to be used
        with models served via `ollama`.
        This way, you can maintain several setting-groups in your .env file,
        one per model type.
        """

        class DynamicConfig(OpenAIGPTConfig):
            pass

        DynamicConfig.Config.env_prefix = prefix.upper() + "_"

        return DynamicConfig


class OpenAIResponse(BaseModel):
    """OpenAI response model, either completion or chat."""

    choices: List[Dict]  # type: ignore
    usage: Dict  # type: ignore


def litellm_logging_fn(model_call_dict: Dict[str, Any]) -> None:
    """Logging function for litellm"""
    try:
        api_input_dict = model_call_dict.get("additional_args", {}).get(
            "complete_input_dict"
        )
        if api_input_dict is not None:
            text = escape(json.dumps(api_input_dict, indent=2))
            print(
                f"[grey37]LITELLM: {text}[/grey37]",
            )
    except Exception:
        pass


# Define a class for OpenAI GPT models that extends the base class
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    client: OpenAI | Groq | Cerebras | None
    async_client: AsyncOpenAI | AsyncGroq | AsyncCerebras | None

    def __init__(self, config: OpenAIGPTConfig = OpenAIGPTConfig()):
        """
        Args:
            config: configuration for openai-gpt model
        """
        # copy the config to avoid modifying the original
        config = config.copy()
        super().__init__(config)
        self.config: OpenAIGPTConfig = config
        # save original model name such as `provider/model` before
        # we strip out the `provider` - we retain the original in
        # case some params are specific to a provider.
        self.chat_model_orig = self.config.chat_model

        # Run the first time the model is used
        self.run_on_first_use = cache(self.config.run_on_first_use)

        # global override of chat_model,
        # to allow quick testing with other models
        if settings.chat_model != "":
            self.config.chat_model = settings.chat_model
            self.chat_model_orig = settings.chat_model
            self.config.completion_model = settings.chat_model

        if len(parts := self.config.chat_model.split("//")) > 1:
            # there is a formatter specified, e.g.
            # "litellm/ollama/mistral//hf" or
            # "local/localhost:8000/v1//mistral-instruct-v0.2"
            formatter = parts[1]
            self.config.chat_model = parts[0]
            if formatter == "hf":
                # e.g. "litellm/ollama/mistral//hf" -> "litellm/ollama/mistral"
                formatter = find_hf_formatter(self.config.chat_model)
                if formatter != "":
                    # e.g. "mistral"
                    self.config.formatter = formatter
                    logging.warning(
                        f"""
                        Using completions (not chat) endpoint with HuggingFace
                        chat_template for {formatter} for
                        model {self.config.chat_model}
                        """
                    )
            else:
                # e.g. "local/localhost:8000/v1//mistral-instruct-v0.2"
                self.config.formatter = formatter

        if self.config.formatter is not None:
            self.config.hf_formatter = HFFormatter(
                HFPromptFormatterConfig(model_name=self.config.formatter)
            )

        self.supports_json_schema: bool = self.config.supports_json_schema or False
        self.supports_strict_tools: bool = self.config.supports_strict_tools or False

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DUMMY_API_KEY)
        self.api_key = config.api_key

        # if model name starts with "litellm",
        # set the actual model name by stripping the "litellm/" prefix
        # and set the litellm flag to True
        if self.config.chat_model.startswith("litellm/") or self.config.litellm:
            # e.g. litellm/ollama/mistral
            self.config.litellm = True
            self.api_base = self.config.api_base
            if self.config.chat_model.startswith("litellm/"):
                # strip the "litellm/" prefix
                # e.g. litellm/ollama/llama2 => ollama/llama2
                self.config.chat_model = self.config.chat_model.split("/", 1)[1]
        elif self.config.chat_model.startswith("local/"):
            # expect this to be of the form "local/localhost:8000/v1",
            # depending on how the model is launched locally.
            # In this case the model served locally behind an OpenAI-compatible API
            # so we can just use `openai.*` methods directly,
            # and don't need a adaptor library like litellm
            self.config.litellm = False
            self.config.seed = None  # some models raise an error when seed is set
            # Extract the api_base from the model name after the "local/" prefix
            self.api_base = self.config.chat_model.split("/", 1)[1]
            if not self.api_base.startswith("http"):
                self.api_base = "http://" + self.api_base
        elif self.config.chat_model.startswith("ollama/"):
            self.config.ollama = True

            # use api_base from config if set, else fall back on OLLAMA_BASE_URL
            self.api_base = self.config.api_base or OLLAMA_BASE_URL
            if self.api_key == OPENAI_API_KEY:
                self.api_key = OLLAMA_API_KEY
            self.config.chat_model = self.config.chat_model.replace("ollama/", "")
        elif self.config.chat_model.startswith("vllm/"):
            self.supports_json_schema = True
            self.config.chat_model = self.config.chat_model.replace("vllm/", "")
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.environ.get("VLLM_API_KEY", DUMMY_API_KEY)
            self.api_base = self.config.api_base or "http://localhost:8000/v1"
            if not self.api_base.startswith("http"):
                self.api_base = "http://" + self.api_base
            if not self.api_base.endswith("/v1"):
                self.api_base = self.api_base + "/v1"
        elif self.config.chat_model.startswith("llamacpp/"):
            self.supports_json_schema = True
            self.api_base = self.config.chat_model.split("/", 1)[1]
            if not self.api_base.startswith("http"):
                self.api_base = "http://" + self.api_base
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.environ.get("LLAMA_API_KEY", DUMMY_API_KEY)
        else:
            self.api_base = self.config.api_base
            # If api_base is unset we use OpenAI's endpoint, which supports
            # these features (with JSON schema restricted to a limited set of models)
            self.supports_strict_tools = self.api_base is None
            self.supports_json_schema = (
                self.api_base is None and self.info().has_structured_output
            )

        if settings.chat_model != "":
            # if we're overriding chat model globally, set completion model to same
            self.config.completion_model = self.config.chat_model

        if self.config.formatter is not None:
            # we want to format chats -> completions using this specific formatter
            self.config.use_completion_for_chat = True
            self.config.completion_model = self.config.chat_model

        if self.config.use_completion_for_chat:
            self.config.use_chat_for_completion = False

        self.is_groq = self.config.chat_model.startswith("groq/")
        self.is_cerebras = self.config.chat_model.startswith("cerebras/")
        self.is_gemini = self.is_gemini_model()
        self.is_deepseek = self.is_deepseek_model()
        self.is_glhf = self.config.chat_model.startswith("glhf/")
        self.is_openrouter = self.config.chat_model.startswith("openrouter/")
        self.is_langdb = self.config.chat_model.startswith("langdb/")
        self.is_litellm_proxy = self.config.chat_model.startswith("litellm-proxy/")

        if self.is_groq:
            # use groq-specific client
            self.config.chat_model = self.config.chat_model.replace("groq/", "")
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.getenv("GROQ_API_KEY", DUMMY_API_KEY)
            self.client = Groq(
                api_key=self.api_key,
            )
            self.async_client = AsyncGroq(
                api_key=self.api_key,
            )
        elif self.is_cerebras:
            # use cerebras-specific client
            self.config.chat_model = self.config.chat_model.replace("cerebras/", "")
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.getenv("CEREBRAS_API_KEY", DUMMY_API_KEY)
            self.client = Cerebras(
                api_key=self.api_key,
            )
            # TODO there is not async client, so should we do anything here?
            self.async_client = AsyncCerebras(
                api_key=self.api_key,
            )
        else:
            # in these cases, there's no specific client: OpenAI python client suffices
            if self.is_litellm_proxy:
                self.config.chat_model = self.config.chat_model.replace(
                    "litellm-proxy/", ""
                )
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = self.config.litellm_proxy.api_key or self.api_key
                self.api_base = self.config.litellm_proxy.api_base or self.api_base
            elif self.is_gemini:
                self.config.chat_model = self.config.chat_model.replace("gemini/", "")
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = os.getenv("GEMINI_API_KEY", DUMMY_API_KEY)
                self.api_base = GEMINI_BASE_URL
            elif self.is_glhf:
                self.config.chat_model = self.config.chat_model.replace("glhf/", "")
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = os.getenv("GLHF_API_KEY", DUMMY_API_KEY)
                self.api_base = GLHF_BASE_URL
            elif self.is_openrouter:
                self.config.chat_model = self.config.chat_model.replace(
                    "openrouter/", ""
                )
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = os.getenv("OPENROUTER_API_KEY", DUMMY_API_KEY)
                self.api_base = OPENROUTER_BASE_URL
            elif self.is_deepseek:
                self.config.chat_model = self.config.chat_model.replace("deepseek/", "")
                self.api_base = DEEPSEEK_BASE_URL
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = os.getenv("DEEPSEEK_API_KEY", DUMMY_API_KEY)
            elif self.is_langdb:
                self.config.chat_model = self.config.chat_model.replace("langdb/", "")
                self.api_base = self.config.langdb_params.base_url
                project_id = self.config.langdb_params.project_id
                if project_id:
                    self.api_base += "/" + project_id + "/v1"
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = self.config.langdb_params.api_key or DUMMY_API_KEY

                if self.config.langdb_params:
                    params = self.config.langdb_params
                    if params.project_id:
                        self.config.headers["x-project-id"] = params.project_id
                    if params.label:
                        self.config.headers["x-label"] = params.label
                    if params.run_id:
                        self.config.headers["x-run-id"] = params.run_id
                    if params.thread_id:
                        self.config.headers["x-thread-id"] = params.thread_id

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                organization=self.config.organization,
                timeout=Timeout(self.config.timeout),
                default_headers=self.config.headers,
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.config.organization,
                base_url=self.api_base,
                timeout=Timeout(self.config.timeout),
                default_headers=self.config.headers,
            )

        self.cache: CacheDB | None = None
        use_cache = self.config.cache_config is not None
        if settings.cache_type == "momento" and use_cache:
            from langroid.cachedb.momento_cachedb import (
                MomentoCache,
                MomentoCacheConfig,
            )

            if config.cache_config is None or not isinstance(
                config.cache_config,
                MomentoCacheConfig,
            ):
                # switch to fresh momento config if needed
                config.cache_config = MomentoCacheConfig()
            self.cache = MomentoCache(config.cache_config)
        elif "redis" in settings.cache_type and use_cache:
            if config.cache_config is None or not isinstance(
                config.cache_config,
                RedisCacheConfig,
            ):
                # switch to fresh redis config if needed
                config.cache_config = RedisCacheConfig(
                    fake="fake" in settings.cache_type
                )
            if "fake" in settings.cache_type:
                # force use of fake redis if global cache_type is "fakeredis"
                config.cache_config.fake = True
            self.cache = RedisCache(config.cache_config)
        elif settings.cache_type != "none" and use_cache:
            raise ValueError(
                f"Invalid cache type {settings.cache_type}. "
                "Valid types are momento, redis, fakeredis, none"
            )

        self.config._validate_litellm()

    def _openai_api_call_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prep the params to be sent to the OpenAI API
        (or any OpenAI-compatible API, e.g. from Ooba or LmStudio)
        for chat-completion.

        Order of priority:
        - (1) Params (mainly max_tokens) in the chat/achat/generate/agenerate call
                (these are passed in via kwargs)
        - (2) Params in OpenAIGPTConfig.params (of class OpenAICallParams)
        - (3) Specific Params in OpenAIGPTConfig (just temperature for now)
        """
        params = dict(
            temperature=self.config.temperature,
        )
        if self.config.params is not None:
            params.update(self.config.params.to_dict_exclude_none())
        params.update(kwargs)
        return params

    def is_openai_chat_model(self) -> bool:
        openai_chat_models = [e.value for e in OpenAIChatModel]
        return self.config.chat_model in openai_chat_models

    def is_openai_completion_model(self) -> bool:
        openai_completion_models = [e.value for e in OpenAICompletionModel]
        return self.config.completion_model in openai_completion_models

    def is_gemini_model(self) -> bool:
        """Are we using the gemini OpenAI-compatible API?"""
        return self.chat_model_orig.startswith("gemini/")

    def is_deepseek_model(self) -> bool:
        deepseek_models = [e.value for e in DeepSeekModel]
        return (
            self.chat_model_orig in deepseek_models
            or self.chat_model_orig.startswith("deepseek/")
        )

    def unsupported_params(self) -> List[str]:
        """
        List of params that are not supported by the current model
        """
        unsupported = set(self.info().unsupported_params)
        for param, model_list in OpenAI_API_ParamInfo().params.items():
            if (
                self.config.chat_model not in model_list
                and self.chat_model_orig not in model_list
            ):
                unsupported.add(param)
        return list(unsupported)

    def rename_params(self) -> Dict[str, str]:
        """
        Map of param name -> new name for specific models.
        Currently main troublemaker is o1* series.
        """
        return self.info().rename_params

    def chat_context_length(self) -> int:
        """
        Context-length for chat-completion models/endpoints.
        Get it from the config if explicitly given,
         otherwise use model_info based on model name, and fall back to
         generic model_info if there's no match.
        """
        return self.config.chat_context_length or self.info().context_length

    def completion_context_length(self) -> int:
        """
        Context-length for completion models/endpoints.
        Get it from the config if explicitly given,
         otherwise use model_info based on model name, and fall back to
         generic model_info if there's no match.
        """
        return (
            self.config.completion_context_length
            or self.completion_info().context_length
        )

    def chat_cost(self) -> Tuple[float, float]:
        """
        (Prompt, Generation) cost per 1000 tokens, for chat-completion
        models/endpoints.
        Get it from the dict, otherwise fail-over to general method
        """
        info = self.info()
        return (info.input_cost_per_million / 1000, info.output_cost_per_million / 1000)

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
        """Get streaming status."""
        return self.config.stream and settings.stream and self.info().allows_streaming

    @no_type_check
    def _process_stream_event(
        self,
        event,
        chat: bool = False,
        tool_deltas: List[Dict[str, Any]] = [],
        has_function: bool = False,
        completion: str = "",
        reasoning: str = "",
        function_args: str = "",
        function_name: str = "",
    ) -> Tuple[bool, bool, str, str, Dict[str, int]]:
        """Process state vars while processing a streaming API response.
            Returns a tuple consisting of:
        - is_break: whether to break out of the loop
        - has_function: whether the response contains a function_call
        - function_name: name of the function
        - function_args: args of the function
        - completion: completion text
        - reasoning: reasoning text
        - usage: usage dict
        """
        # convert event obj (of type ChatCompletionChunk) to dict so rest of code,
        # which expects dicts, works as it did before switching to openai v1.x
        if not isinstance(event, dict):
            event = event.model_dump()

        usage = event.get("usage", {}) or {}
        choices = event.get("choices", [{}])
        if choices is None or len(choices) == 0:
            choices = [{}]
        if len(usage) > 0 and len(choices[0]) == 0:
            # we have a "usage" chunk, and empty choices, so we're done
            # ASSUMPTION: a usage chunk ONLY arrives AFTER all normal completion text!
            # If any API does not follow this, we need to change this code.
            return (
                True,
                has_function,
                function_name,
                function_args,
                completion,
                reasoning,
                usage,
            )
        event_args = ""
        event_fn_name = ""
        event_tool_deltas: Optional[List[Dict[str, Any]]] = None
        silent = settings.quiet
        # The first two events in the stream of Azure OpenAI is useless.
        # In the 1st: choices list is empty, in the 2nd: the dict delta has null content
        if chat:
            delta = choices[0].get("delta", {})
            # capture both content and reasoning_content
            event_text = delta.get("content", "")
            event_reasoning = delta.get(
                "reasoning_content",
                delta.get("reasoning", ""),
            )
            if "function_call" in delta and delta["function_call"] is not None:
                if "name" in delta["function_call"]:
                    event_fn_name = delta["function_call"]["name"]
                if "arguments" in delta["function_call"]:
                    event_args = delta["function_call"]["arguments"]
            if "tool_calls" in delta and delta["tool_calls"] is not None:
                # it's a list of deltas, usually just one
                event_tool_deltas = delta["tool_calls"]
                tool_deltas += event_tool_deltas
        else:
            event_text = choices[0]["text"]
            event_reasoning = ""  # TODO: Ignoring reasoning for non-chat models

        finish_reason = choices[0].get("finish_reason", "")
        if not event_text and finish_reason == "content_filter":
            filter_names = [
                n
                for n, r in choices[0].get("content_filter_results", {}).items()
                if r.get("filtered")
            ]
            event_text = (
                "Cannot respond due to content filters ["
                + ", ".join(filter_names)
                + "]"
            )
            logging.warning("LLM API returned content filter error: " + event_text)

        if event_text:
            completion += event_text
            if not silent:
                sys.stdout.write(Colors().GREEN + event_text)
                sys.stdout.flush()
            self.config.streamer(event_text, StreamEventType.TEXT)
        if event_reasoning:
            reasoning += event_reasoning
            if not silent:
                sys.stdout.write(Colors().GREEN_DIM + event_reasoning)
                sys.stdout.flush()
            self.config.streamer(event_reasoning, StreamEventType.TEXT)
        if event_fn_name:
            function_name = event_fn_name
            has_function = True
            if not silent:
                sys.stdout.write(Colors().GREEN + "FUNC: " + event_fn_name + ": ")
                sys.stdout.flush()
            self.config.streamer(event_fn_name, StreamEventType.FUNC_NAME)

        if event_args:
            function_args += event_args
            if not silent:
                sys.stdout.write(Colors().GREEN + event_args)
                sys.stdout.flush()
            self.config.streamer(event_args, StreamEventType.FUNC_ARGS)

        if event_tool_deltas is not None:
            # print out streaming tool calls, if not async
            for td in event_tool_deltas:
                if td["function"]["name"] is not None:
                    tool_fn_name = td["function"]["name"]
                    if not silent:
                        sys.stdout.write(
                            Colors().GREEN + "OAI-TOOL: " + tool_fn_name + ": "
                        )
                        sys.stdout.flush()
                    self.config.streamer(tool_fn_name, StreamEventType.TOOL_NAME)
                if td["function"]["arguments"] != "":
                    tool_fn_args = td["function"]["arguments"]
                    if not silent:
                        sys.stdout.write(Colors().GREEN + tool_fn_args)
                        sys.stdout.flush()
                    self.config.streamer(tool_fn_args, StreamEventType.TOOL_ARGS)

        # show this delta in the stream
        is_break = finish_reason in [
            "stop",
            "function_call",
            "tool_calls",
        ]
        # for function_call, finish_reason does not necessarily
        # contain "function_call" as mentioned in the docs.
        # So we check for "stop" or "function_call" here.
        return (
            is_break,
            has_function,
            function_name,
            function_args,
            completion,
            reasoning,
            usage,
        )

    @no_type_check
    async def _process_stream_event_async(
        self,
        event,
        chat: bool = False,
        tool_deltas: List[Dict[str, Any]] = [],
        has_function: bool = False,
        completion: str = "",
        reasoning: str = "",
        function_args: str = "",
        function_name: str = "",
    ) -> Tuple[bool, bool, str, str]:
        """Process state vars while processing a streaming API response.
            Returns a tuple consisting of:
        - is_break: whether to break out of the loop
        - has_function: whether the response contains a function_call
        - function_name: name of the function
        - function_args: args of the function
        - completion: completion text
        - reasoning: reasoning text
        - usage: usage dict
        """
        # convert event obj (of type ChatCompletionChunk) to dict so rest of code,
        # which expects dicts, works as it did before switching to openai v1.x
        if not isinstance(event, dict):
            event = event.model_dump()

        usage = event.get("usage", {}) or {}
        choices = event.get("choices", [{}])
        if len(choices) == 0:
            choices = [{}]
        if len(usage) > 0 and len(choices[0]) == 0:
            # we got usage chunk, and empty choices, so we're done
            return (
                True,
                has_function,
                function_name,
                function_args,
                completion,
                reasoning,
                usage,
            )
        event_args = ""
        event_fn_name = ""
        event_tool_deltas: Optional[List[Dict[str, Any]]] = None
        silent = self.config.async_stream_quiet or settings.quiet
        # The first two events in the stream of Azure OpenAI is useless.
        # In the 1st: choices list is empty, in the 2nd: the dict delta has null content
        if chat:
            delta = choices[0].get("delta", {})
            event_text = delta.get("content", "")
            event_reasoning = delta.get(
                "reasoning_content",
                delta.get("reasoning", ""),
            )
            if "function_call" in delta and delta["function_call"] is not None:
                if "name" in delta["function_call"]:
                    event_fn_name = delta["function_call"]["name"]
                if "arguments" in delta["function_call"]:
                    event_args = delta["function_call"]["arguments"]
            if "tool_calls" in delta and delta["tool_calls"] is not None:
                # it's a list of deltas, usually just one
                event_tool_deltas = delta["tool_calls"]
                tool_deltas += event_tool_deltas
        else:
            event_text = choices[0]["text"]
            event_reasoning = ""  # TODO: Ignoring reasoning for non-chat models
        if event_text:
            completion += event_text
            if not silent:
                sys.stdout.write(Colors().GREEN + event_text)
                sys.stdout.flush()
            await self.config.streamer_async(event_text, StreamEventType.TEXT)
        if event_reasoning:
            reasoning += event_reasoning
            if not silent:
                sys.stdout.write(Colors().GREEN + event_reasoning)
                sys.stdout.flush()
            await self.config.streamer_async(event_reasoning, StreamEventType.TEXT)
        if event_fn_name:
            function_name = event_fn_name
            has_function = True
            if not silent:
                sys.stdout.write(Colors().GREEN + "FUNC: " + event_fn_name + ": ")
                sys.stdout.flush()
            await self.config.streamer_async(event_fn_name, StreamEventType.FUNC_NAME)

        if event_args:
            function_args += event_args
            if not silent:
                sys.stdout.write(Colors().GREEN + event_args)
                sys.stdout.flush()
            await self.config.streamer_async(event_args, StreamEventType.FUNC_ARGS)

        if event_tool_deltas is not None:
            # print out streaming tool calls, if not async
            for td in event_tool_deltas:
                if td["function"]["name"] is not None:
                    tool_fn_name = td["function"]["name"]
                    if not silent:
                        sys.stdout.write(
                            Colors().GREEN + "OAI-TOOL: " + tool_fn_name + ": "
                        )
                        sys.stdout.flush()
                    await self.config.streamer_async(
                        tool_fn_name, StreamEventType.TOOL_NAME
                    )
                if td["function"]["arguments"] != "":
                    tool_fn_args = td["function"]["arguments"]
                    if not silent:
                        sys.stdout.write(Colors().GREEN + tool_fn_args)
                        sys.stdout.flush()
                    await self.config.streamer_async(
                        tool_fn_args, StreamEventType.TOOL_ARGS
                    )

        # show this delta in the stream
        is_break = choices[0].get("finish_reason", "") in [
            "stop",
            "function_call",
            "tool_calls",
        ]
        # for function_call, finish_reason does not necessarily
        # contain "function_call" as mentioned in the docs.
        # So we check for "stop" or "function_call" here.
        return (
            is_break,
            has_function,
            function_name,
            function_args,
            completion,
            reasoning,
            usage,
        )

    @retry_with_exponential_backoff
    def _stream_response(  # type: ignore
        self, response, chat: bool = False
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        """
        Grab and print streaming response from API.
        Args:
            response: event-sequence emitted by API
            chat: whether in chat-mode (or else completion-mode)
        Returns:
            Tuple consisting of:
                LLMResponse object (with message, usage),
                Dict version of OpenAIResponse object (with choices, usage)

        """
        completion = ""
        reasoning = ""
        function_args = ""
        function_name = ""

        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()
        has_function = False
        tool_deltas: List[Dict[str, Any]] = []
        token_usage: Dict[str, int] = {}
        done: bool = False
        try:
            for event in response:
                (
                    is_break,
                    has_function,
                    function_name,
                    function_args,
                    completion,
                    reasoning,
                    usage,
                ) = self._process_stream_event(
                    event,
                    chat=chat,
                    tool_deltas=tool_deltas,
                    has_function=has_function,
                    completion=completion,
                    reasoning=reasoning,
                    function_args=function_args,
                    function_name=function_name,
                )
                if len(usage) > 0:
                    # capture the token usage when non-empty
                    token_usage = usage
                if is_break:
                    if not self.get_stream() or done:
                        # if not streaming, then we don't wait for last "usage" chunk
                        break
                    else:
                        # mark done, so we quit after the last "usage" chunk
                        done = True

        except Exception as e:
            logging.warning("Error while processing stream response: %s", str(e))

        if not settings.quiet:
            print("")
        # TODO- get usage info in stream mode (?)

        return self._create_stream_response(
            chat=chat,
            tool_deltas=tool_deltas,
            has_function=has_function,
            completion=completion,
            reasoning=reasoning,
            function_args=function_args,
            function_name=function_name,
            usage=token_usage,
        )

    @async_retry_with_exponential_backoff
    async def _stream_response_async(  # type: ignore
        self, response, chat: bool = False
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
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
        reasoning = ""
        function_args = ""
        function_name = ""

        sys.stdout.write(Colors().GREEN)
        sys.stdout.flush()
        has_function = False
        tool_deltas: List[Dict[str, Any]] = []
        token_usage: Dict[str, int] = {}
        done: bool = False
        try:
            async for event in response:
                (
                    is_break,
                    has_function,
                    function_name,
                    function_args,
                    completion,
                    reasoning,
                    usage,
                ) = await self._process_stream_event_async(
                    event,
                    chat=chat,
                    tool_deltas=tool_deltas,
                    has_function=has_function,
                    completion=completion,
                    reasoning=reasoning,
                    function_args=function_args,
                    function_name=function_name,
                )
                if len(usage) > 0:
                    # capture the token usage when non-empty
                    token_usage = usage
                if is_break:
                    if not self.get_stream() or done:
                        # if not streaming, then we don't wait for last "usage" chunk
                        break
                    else:
                        # mark done, so we quit after the next "usage" chunk
                        done = True

        except Exception as e:
            logging.warning("Error while processing stream response: %s", str(e))

        if not settings.quiet:
            print("")
        # TODO- get usage info in stream mode (?)

        return self._create_stream_response(
            chat=chat,
            tool_deltas=tool_deltas,
            has_function=has_function,
            completion=completion,
            reasoning=reasoning,
            function_args=function_args,
            function_name=function_name,
            usage=token_usage,
        )

    @staticmethod
    def tool_deltas_to_tools(
        tools: List[Dict[str, Any]],
    ) -> Tuple[
        str,
        List[OpenAIToolCall],
        List[Dict[str, Any]],
    ]:
        """
        Convert accumulated tool-call deltas to OpenAIToolCall objects.
        Adapted from this excellent code:
         https://community.openai.com/t/help-for-function-calls-with-streaming/627170/2

        Args:
            tools: list of tool deltas received from streaming API

        Returns:
            str: plain text corresponding to tool calls that failed to parse
            List[OpenAIToolCall]: list of OpenAIToolCall objects
            List[Dict[str, Any]]: list of tool dicts
                (to reconstruct OpenAI API response, so it can be cached)
        """
        # Initialize a dictionary with default values

        # idx -> dict repr of tool
        # (used to simulate OpenAIResponse object later, and also to
        # accumulate function args as strings)
        idx2tool_dict: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "id": None,
                "function": {"arguments": "", "name": None},
                "type": None,
            }
        )

        for tool_delta in tools:
            if tool_delta["id"] is not None:
                idx2tool_dict[tool_delta["index"]]["id"] = tool_delta["id"]

            if tool_delta["function"]["name"] is not None:
                idx2tool_dict[tool_delta["index"]]["function"]["name"] = tool_delta[
                    "function"
                ]["name"]

            idx2tool_dict[tool_delta["index"]]["function"]["arguments"] += tool_delta[
                "function"
            ]["arguments"]

            if tool_delta["type"] is not None:
                idx2tool_dict[tool_delta["index"]]["type"] = tool_delta["type"]

        # (try to) parse the fn args of each tool
        contents: List[str] = []
        good_indices = []
        id2args: Dict[str, None | Dict[str, Any]] = {}
        for idx, tool_dict in idx2tool_dict.items():
            failed_content, args_dict = OpenAIGPT._parse_function_args(
                tool_dict["function"]["arguments"]
            )
            # used to build tool_calls_list below
            id2args[tool_dict["id"]] = args_dict or None  # if {}, store as None
            if failed_content != "":
                contents.append(failed_content)
            else:
                good_indices.append(idx)

        # remove the failed tool calls
        idx2tool_dict = {
            idx: tool_dict
            for idx, tool_dict in idx2tool_dict.items()
            if idx in good_indices
        }

        # create OpenAIToolCall list
        tool_calls_list = [
            OpenAIToolCall(
                id=tool_dict["id"],
                function=LLMFunctionCall(
                    name=tool_dict["function"]["name"],
                    arguments=id2args.get(tool_dict["id"]),
                ),
                type=tool_dict["type"],
            )
            for tool_dict in idx2tool_dict.values()
        ]
        return "\n".join(contents), tool_calls_list, list(idx2tool_dict.values())

    @staticmethod
    def _parse_function_args(args: str) -> Tuple[str, Dict[str, Any]]:
        """
        Try to parse the `args` string as function args.

        Args:
            args: string containing function args

        Returns:
            Tuple of content, function name and args dict.
            If parsing unsuccessful, returns the original string as content,
            else returns the args dict.
        """
        content = ""
        args_dict = {}
        try:
            stripped_fn_args = args.strip()
            dict_or_list = parse_imperfect_json(stripped_fn_args)
            if not isinstance(dict_or_list, dict):
                raise ValueError(
                    f"""
                        Invalid function args: {stripped_fn_args}
                        parsed as {dict_or_list},
                        which is not a valid dict.
                        """
                )
            args_dict = dict_or_list
        except (SyntaxError, ValueError) as e:
            logging.warning(
                f"""
                    Parsing OpenAI function args failed: {args};
                    treating args as normal message. Error detail:
                    {e}
                    """
            )
            content = args

        return content, args_dict

    def _create_stream_response(
        self,
        chat: bool = False,
        tool_deltas: List[Dict[str, Any]] = [],
        has_function: bool = False,
        completion: str = "",
        reasoning: str = "",
        function_args: str = "",
        function_name: str = "",
        usage: Dict[str, int] = {},
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        """
        Create an LLMResponse object from the streaming API response.

        Args:
            chat: whether in chat-mode (or else completion-mode)
            tool_deltas: list of tool deltas received from streaming API
            has_function: whether the response contains a function_call
            completion: completion text
            reasoning: reasoning text
            function_args: string representing function args
            function_name: name of the function
            usage: token usage dict
        Returns:
            Tuple consisting of:
                LLMResponse object (with message, usage),
                Dict version of OpenAIResponse object (with choices, usage)
                    (this is needed so we can cache the response, as if it were
                    a non-streaming response)
        """
        # check if function_call args are valid, if not,
        # treat this as a normal msg, not a function call
        args: Dict[str, Any] = {}
        if has_function and function_args != "":
            content, args = self._parse_function_args(function_args)
            completion = completion + content
            if content != "":
                has_function = False

        # mock openai response so we can cache it
        if chat:
            failed_content, tool_calls, tool_dicts = OpenAIGPT.tool_deltas_to_tools(
                tool_deltas,
            )
            completion = completion + "\n" + failed_content
            msg: Dict[str, Any] = dict(
                message=dict(
                    content=completion,
                    reasoning_content=reasoning,
                ),
            )
            if len(tool_dicts) > 0:
                msg["message"]["tool_calls"] = tool_dicts

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
            # TODO: Ignoring reasoning content for non-chat models

        # create an OpenAIResponse object so we can cache it as if it were
        # a non-streaming response
        openai_response = OpenAIResponse(
            choices=[msg],
            usage=dict(total_tokens=0),
        )
        if reasoning == "":
            # some LLM APIs may not return a separate reasoning field,
            # and the reasoning may be included in the message content
            # within delimiters like <think> ... </think>
            reasoning, completion = self.get_reasoning_final(completion)
        return (
            LLMResponse(
                message=completion,
                reasoning=reasoning,
                cached=False,
                # don't allow empty list [] here
                oai_tool_calls=tool_calls or None if len(tool_deltas) > 0 else None,
                function_call=function_call if has_function else None,
                usage=LLMTokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost=self._cost_chat_model(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    ),
                ),
            ),
            openai_response.dict(),
        )

    def _cache_store(self, k: str, v: Any) -> None:
        if self.cache is None:
            return
        try:
            self.cache.store(k, v)
        except Exception as e:
            logging.error(f"Error in OpenAIGPT._cache_store: {e}")
            pass

    def _cache_lookup(self, fn_name: str, **kwargs: Dict[str, Any]) -> Tuple[str, Any]:
        if self.cache is None:
            return "", None  # no cache, return empty key and None result
        # Use the kwargs as the cache key
        sorted_kwargs_str = str(sorted(kwargs.items()))
        raw_key = f"{fn_name}:{sorted_kwargs_str}"

        # Hash the key to a fixed length using SHA256
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

        if not settings.cache:
            # when caching disabled, return the hashed_key and none result
            return hashed_key, None
        # Try to get the result from the cache
        try:
            cached_val = self.cache.retrieve(hashed_key)
        except Exception as e:
            logging.error(f"Error in OpenAIGPT._cache_lookup: {e}")
            return hashed_key, None
        return hashed_key, cached_val

    def _cost_chat_model(self, prompt: int, completion: int) -> float:
        price = self.chat_cost()
        return (price[0] * prompt + price[1] * completion) / 1000

    def _get_non_stream_token_usage(
        self, cached: bool, response: Dict[str, Any]
    ) -> LLMTokenUsage:
        """
        Extracts token usage from ``response`` and computes cost, only when NOT
        in streaming mode, since the LLM API (OpenAI currently) was not
        populating the usage fields in streaming mode (but as of Sep 2024, streaming
        responses include  usage info as well, so we should update the code
        to directly use usage information from the streaming response, which is more
        accurate, esp with "thinking" LLMs like o1 series which consume
        thinking tokens).
        In streaming mode, these are set to zero for
        now, and will be updated later by the fn ``update_token_usage``.
        """
        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        if not cached and not self.get_stream() and response["usage"] is not None:
            prompt_tokens = response["usage"]["prompt_tokens"] or 0
            completion_tokens = response["usage"]["completion_tokens"] or 0
            cost = self._cost_chat_model(prompt_tokens, completion_tokens)

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        self.run_on_first_use()

        try:
            return self._generate(prompt, max_tokens)
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.generate: "))
            raise e

    def _generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        if self.config.use_chat_for_completion:
            return self.chat(messages=prompt, max_tokens=max_tokens)

        if self.is_groq or self.is_cerebras:
            raise ValueError("Groq, Cerebras do not support pure completions")

        if settings.debug:
            print(f"[grey37]PROMPT: {escape(prompt)}[/grey37]")

        @retry_with_exponential_backoff
        def completions_with_backoff(**kwargs):  # type: ignore
            cached = False
            hashed_key, result = self._cache_lookup("Completion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    print("[grey37]CACHED[/grey37]")
            else:
                if self.config.litellm:
                    from litellm import completion as litellm_completion

                    completion_call = litellm_completion
                else:
                    if self.client is None:
                        raise ValueError(
                            "OpenAI/equivalent chat-completion client not set"
                        )
                    assert isinstance(self.client, OpenAI)
                    completion_call = self.client.completions.create
                if self.config.litellm and settings.debug:
                    kwargs["logger_fn"] = litellm_logging_fn
                # If it's not in the cache, call the API
                result = completion_call(**kwargs)
                if self.get_stream():
                    llm_response, openai_response = self._stream_response(
                        result,
                        chat=self.config.litellm,
                    )
                    self._cache_store(hashed_key, openai_response)
                    return cached, hashed_key, openai_response
                else:
                    self._cache_store(hashed_key, result.model_dump())
            return cached, hashed_key, result

        kwargs: Dict[str, Any] = dict(model=self.config.completion_model)
        if self.config.litellm:
            # TODO this is a temp fix, we should really be using a proper completion fn
            # that takes a pre-formatted prompt, rather than mocking it as a sys msg.
            kwargs["messages"] = [dict(content=prompt, role=Role.SYSTEM)]
        else:  # any other OpenAI-compatible endpoint
            kwargs["prompt"] = prompt
        args = dict(
            **kwargs,
            max_tokens=max_tokens,  # for output/completion
            stream=self.get_stream(),
        )
        args = self._openai_api_call_params(args)
        cached, hashed_key, response = completions_with_backoff(**args)
        # assume response is an actual response rather than a streaming event
        if not isinstance(response, dict):
            response = response.dict()
        if "message" in response["choices"][0]:
            msg = response["choices"][0]["message"]["content"].strip()
        else:
            msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, cached=cached)

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        self.run_on_first_use()

        try:
            return await self._agenerate(prompt, max_tokens)
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.agenerate: "))
            raise e

    async def _agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        # note we typically will not have self.config.stream = True
        # when issuing several api calls concurrently/asynchronously.
        # The calling fn should use the context `with Streaming(..., False)` to
        # disable streaming.
        if self.config.use_chat_for_completion:
            return await self.achat(messages=prompt, max_tokens=max_tokens)

        if self.is_groq or self.is_cerebras:
            raise ValueError("Groq, Cerebras do not support pure completions")

        if settings.debug:
            print(f"[grey37]PROMPT: {escape(prompt)}[/grey37]")

        # WARNING: .Completion.* endpoints are deprecated,
        # and as of Sep 2023 only legacy models will work here,
        # e.g. text-davinci-003, text-ada-001.
        @async_retry_with_exponential_backoff
        async def completions_with_backoff(**kwargs):  # type: ignore
            cached = False
            hashed_key, result = self._cache_lookup("AsyncCompletion", **kwargs)
            if result is not None:
                cached = True
                if settings.debug:
                    print("[grey37]CACHED[/grey37]")
            else:
                if self.config.litellm:
                    from litellm import acompletion as litellm_acompletion
                # TODO this may not work: text_completion is not async,
                # and we didn't find an async version in litellm
                assert isinstance(self.async_client, AsyncOpenAI)
                acompletion_call = (
                    litellm_acompletion
                    if self.config.litellm
                    else self.async_client.completions.create
                )
                if self.config.litellm and settings.debug:
                    kwargs["logger_fn"] = litellm_logging_fn
                # If it's not in the cache, call the API
                result = await acompletion_call(**kwargs)
                self._cache_store(hashed_key, result.model_dump())
            return cached, hashed_key, result

        kwargs: Dict[str, Any] = dict(model=self.config.completion_model)
        if self.config.litellm:
            # TODO this is a temp fix, we should really be using a proper completion fn
            # that takes a pre-formatted prompt, rather than mocking it as a sys msg.
            kwargs["messages"] = [dict(content=prompt, role=Role.SYSTEM)]
        else:  # any other OpenAI-compatible endpoint
            kwargs["prompt"] = prompt
        cached, hashed_key, response = await completions_with_backoff(
            **kwargs,
            max_tokens=max_tokens,
            stream=False,
        )
        # assume response is an actual response rather than a streaming event
        if not isinstance(response, dict):
            response = response.dict()
        if "message" in response["choices"][0]:
            msg = response["choices"][0]["message"]["content"].strip()
        else:
            msg = response["choices"][0]["text"].strip()
        return LLMResponse(message=msg, cached=cached)

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        self.run_on_first_use()

        if self.config.use_completion_for_chat and not self.is_openai_chat_model():
            # only makes sense for non-OpenAI models
            if self.config.formatter is None or self.config.hf_formatter is None:
                raise ValueError(
                    """
                    `formatter` must be specified in config to use completion for chat.
                    """
                )
            if isinstance(messages, str):
                messages = [
                    LLMMessage(
                        role=Role.SYSTEM, content="You are a helpful assistant."
                    ),
                    LLMMessage(role=Role.USER, content=messages),
                ]
            prompt = self.config.hf_formatter.format(messages)
            return self.generate(prompt=prompt, max_tokens=max_tokens)
        try:
            return self._chat(
                messages,
                max_tokens,
                tools,
                tool_choice,
                functions,
                function_call,
                response_format,
            )
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.chat: "))
            raise e

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        self.run_on_first_use()

        # turn off streaming for async calls
        if (
            self.config.use_completion_for_chat
            and not self.is_openai_chat_model()
            and not self.is_openai_completion_model()
        ):
            # only makes sense for local models, where we are trying to
            # convert a chat dialog msg-sequence to a simple completion prompt.
            if self.config.formatter is None:
                raise ValueError(
                    """
                    `formatter` must be specified in config to use completion for chat.
                    """
                )
            formatter = HFFormatter(
                HFPromptFormatterConfig(model_name=self.config.formatter)
            )
            if isinstance(messages, str):
                messages = [
                    LLMMessage(
                        role=Role.SYSTEM, content="You are a helpful assistant."
                    ),
                    LLMMessage(role=Role.USER, content=messages),
                ]
            prompt = formatter.format(messages)
            return await self.agenerate(prompt=prompt, max_tokens=max_tokens)
        try:
            result = await self._achat(
                messages,
                max_tokens,
                tools,
                tool_choice,
                functions,
                function_call,
                response_format,
            )
            return result
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.achat: "))
            raise e

    @retry_with_exponential_backoff
    def _chat_completions_with_backoff(self, **kwargs):  # type: ignore
        cached = False
        hashed_key, result = self._cache_lookup("Completion", **kwargs)
        if result is not None:
            cached = True
            if settings.debug:
                print("[grey37]CACHED[/grey37]")
        else:
            # If it's not in the cache, call the API
            if self.config.litellm:
                from litellm import completion as litellm_completion

                completion_call = litellm_completion
            else:
                if self.client is None:
                    raise ValueError("OpenAI/equivalent chat-completion client not set")
                completion_call = self.client.chat.completions.create
            if self.config.litellm and settings.debug:
                kwargs["logger_fn"] = litellm_logging_fn
            result = completion_call(**kwargs)

            if self.get_stream():
                # If streaming, cannot cache result
                # since it is a generator. Instead,
                # we hold on to the hashed_key and
                # cache the result later

                # Test if this is a stream with an exception by
                # trying to get first chunk: Some providers like LiteLLM
                # produce a valid stream object `result` instead of throwing a
                # rate-limit error, and if we don't catch it here,
                # we end up returning an empty response and not
                # using the retry mechanism in the decorator.
                try:
                    # try to get the first chunk to check for errors
                    test_iter = iter(result)
                    first_chunk = next(test_iter)
                    # If we get here without error, recreate the stream
                    result = chain([first_chunk], test_iter)
                except StopIteration:
                    # Empty stream is fine
                    pass
                except Exception as e:
                    # Propagate any errors in the stream
                    raise e
            else:
                self._cache_store(hashed_key, result.model_dump())
        return cached, hashed_key, result

    @async_retry_with_exponential_backoff
    async def _achat_completions_with_backoff(self, **kwargs):  # type: ignore
        cached = False
        hashed_key, result = self._cache_lookup("Completion", **kwargs)
        if result is not None:
            cached = True
            if settings.debug:
                print("[grey37]CACHED[/grey37]")
        else:
            if self.config.litellm:
                from litellm import acompletion as litellm_acompletion

                acompletion_call = litellm_acompletion
            else:
                if self.async_client is None:
                    raise ValueError(
                        "OpenAI/equivalent async chat-completion client not set"
                    )
                acompletion_call = self.async_client.chat.completions.create
            if self.config.litellm and settings.debug:
                kwargs["logger_fn"] = litellm_logging_fn
            # If it's not in the cache, call the API
            result = await acompletion_call(**kwargs)
            if self.get_stream():
                try:
                    # Try to peek at the first chunk to immediately catch any errors
                    # Store the original result (the stream)
                    original_stream = result

                    # Manually create and advance the iterator to check for errors
                    stream_iter = original_stream.__aiter__()
                    try:
                        # This will raise an exception if the stream is invalid
                        first_chunk = await anext(stream_iter)

                        # If we reach here, the stream started successfully
                        # Now recreate a fresh stream from the original API result
                        # Otherwise, return a new stream that yields the first chunk
                        # and remaining items
                        async def combined_stream():  # type: ignore
                            yield first_chunk
                            async for chunk in stream_iter:
                                yield chunk

                        result = combined_stream()  # type: ignore
                    except StopAsyncIteration:
                        # Empty stream is normal - nothing to do
                        pass
                except Exception as e:
                    # Any exception here should be raised to trigger the retry mechanism
                    raise e
            else:
                self._cache_store(hashed_key, result.model_dump())
        return cached, hashed_key, result

    def _prep_chat_completion(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> Dict[str, Any]:
        """Prepare args for LLM chat-completion API call"""
        if isinstance(messages, str):
            llm_messages = [
                LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
                LLMMessage(role=Role.USER, content=messages),
            ]
        else:
            llm_messages = messages
            if (
                len(llm_messages) == 1
                and llm_messages[0].role == Role.SYSTEM
                # TODO: we will unconditionally insert a dummy user msg
                # if the only msg is a system msg.
                # We could make this conditional on ModelInfo.needs_first_user_message
            ):
                # some LLMs, notable Gemini as of 12/11/24,
                # require the first message to be from the user,
                # so insert a dummy user msg if needed.
                llm_messages.insert(
                    1,
                    LLMMessage(
                        role=Role.USER, content="Follow the above instructions."
                    ),
                )

        chat_model = self.config.chat_model

        args: Dict[str, Any] = dict(
            model=chat_model,
            messages=[
                m.api_dict(has_system_role=self.info().allows_system_message)
                for m in (llm_messages)
            ],
            max_tokens=max_tokens,
            stream=self.get_stream(),
        )
        if self.get_stream():
            args.update(
                dict(
                    # get token-usage numbers in stream mode from OpenAI API,
                    # and possibly other OpenAI-compatible APIs.
                    stream_options=dict(include_usage=True),
                )
            )
        args.update(self._openai_api_call_params(args))
        # only include functions-related args if functions are provided
        # since the OpenAI API will throw an error if `functions` is None or []
        if functions is not None:
            args.update(
                dict(
                    functions=[f.dict() for f in functions],
                    function_call=function_call,
                )
            )
        if tools is not None:
            if self.config.parallel_tool_calls is not None:
                args["parallel_tool_calls"] = self.config.parallel_tool_calls

            if any(t.strict for t in tools) and (
                self.config.parallel_tool_calls is None
                or self.config.parallel_tool_calls
            ):
                parallel_strict_warning()
            args.update(
                dict(
                    tools=[
                        dict(
                            type="function",
                            function=t.function.dict()
                            | ({"strict": t.strict} if t.strict is not None else {}),
                        )
                        for t in tools
                    ],
                    tool_choice=tool_choice,
                )
            )
        if response_format is not None:
            args["response_format"] = response_format.to_dict()

        for p in self.unsupported_params():
            # some models e.g. o1-mini (as of sep 2024) don't support some params,
            # like temperature and stream, so we need to remove them.
            args.pop(p, None)

        param_rename_map = self.rename_params()
        for old_param, new_param in param_rename_map.items():
            if old_param in args:
                args[new_param] = args.pop(old_param)

        # finally, get rid of extra_body params exclusive to certain models
        extra_params = args.get("extra_body", {})
        if extra_params:
            for param, model_list in OpenAI_API_ParamInfo().extra_parameters.items():
                if (
                    self.config.chat_model not in model_list
                    and self.chat_model_orig not in model_list
                ):
                    extra_params.pop(param, None)
            if extra_params:
                args["extra_body"] = extra_params
        return args

    def _process_chat_completion_response(
        self,
        cached: bool,
        response: Dict[str, Any],
    ) -> LLMResponse:
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
                    "reasoning_content": "Okay, let's see here, hmmm...",
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
        if response.get("choices") is None:
            message = {}
        else:
            message = response["choices"][0].get("message", {})
        if message is None:
            message = {}
        msg = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        if reasoning == "" and msg is not None:
            # some LLM APIs may not return a separate reasoning field,
            # and the reasoning may be included in the message content
            # within delimiters like <think> ... </think>
            reasoning, msg = self.get_reasoning_final(msg)

        if message.get("function_call") is None:
            fun_call = None
        else:
            try:
                fun_call = LLMFunctionCall.from_dict(message["function_call"])
            except (ValueError, SyntaxError):
                logging.warning(
                    "Could not parse function arguments: "
                    f"{message['function_call']['arguments']} "
                    f"for function {message['function_call']['name']} "
                    "treating as normal non-function message"
                )
                fun_call = None
                args_str = message["function_call"]["arguments"] or ""
                msg_str = message["content"] or ""
                msg = msg_str + args_str
        oai_tool_calls = None
        if message.get("tool_calls") is not None:
            oai_tool_calls = []
            for tool_call_dict in message["tool_calls"]:
                try:
                    tool_call = OpenAIToolCall.from_dict(tool_call_dict)
                    oai_tool_calls.append(tool_call)
                except (ValueError, SyntaxError):
                    logging.warning(
                        "Could not parse tool call: "
                        f"{json.dumps(tool_call_dict)} "
                        "treating as normal non-tool message"
                    )
                    msg = msg + "\n" + json.dumps(tool_call_dict)
        return LLMResponse(
            message=msg.strip() if msg is not None else "",
            reasoning=reasoning.strip() if reasoning is not None else "",
            function_call=fun_call,
            oai_tool_calls=oai_tool_calls or None,  # don't allow empty list [] here
            cached=cached,
            usage=self._get_non_stream_token_usage(cached, response),
        )

    def _chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
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
        args = self._prep_chat_completion(
            messages,
            max_tokens,
            tools,
            tool_choice,
            functions,
            function_call,
            response_format,
        )
        cached, hashed_key, response = self._chat_completions_with_backoff(**args)
        if self.get_stream() and not cached:
            llm_response, openai_response = self._stream_response(response, chat=True)
            self._cache_store(hashed_key, openai_response)
            return llm_response  # type: ignore
        if isinstance(response, dict):
            response_dict = response
        else:
            response_dict = response.model_dump()
        return self._process_chat_completion_response(cached, response_dict)

    async def _achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        """
        Async version of _chat(). See that function for details.
        """
        args = self._prep_chat_completion(
            messages,
            max_tokens,
            tools,
            tool_choice,
            functions,
            function_call,
            response_format,
        )
        cached, hashed_key, response = await self._achat_completions_with_backoff(
            **args
        )
        if self.get_stream() and not cached:
            llm_response, openai_response = await self._stream_response_async(
                response, chat=True
            )
            self._cache_store(hashed_key, openai_response)
            return llm_response  # type: ignore
        if isinstance(response, dict):
            response_dict = response
        else:
            response_dict = response.model_dump()
        return self._process_chat_completion_response(cached, response_dict)
