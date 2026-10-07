import hashlib
import json
import logging
import os
import re
import sys
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import cache
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import openai
from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from groq import AsyncGroq, Groq
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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
from langroid.language_models.client_cache import (
    get_async_cerebras_client,
    get_async_groq_client,
    get_async_openai_client,
    get_cerebras_client,
    get_groq_client,
    get_openai_client,
    wrap_api_key_provider_async,
)
from langroid.language_models.config import HFPromptFormatterConfig
from langroid.language_models.httpx_compat import (
    Timeout,
    import_httpx,
    missing_httpx_message,
)
from langroid.language_models.model_info import (
    DeepSeekModel,
    MiniMaxModel,
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
from langroid.language_models.provider_params import (
    DUMMY_API_KEY,
    LangDBParams,
    PortkeyParams,
)
from langroid.language_models.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    get_rate_limiter,
    rate_limit_error_headers,
)
from langroid.language_models.utils import (
    async_retry_with_exponential_backoff,
    retry_with_exponential_backoff,
)
from langroid.parsing.parse_json import parse_imperfect_json
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
# Provider prefixes that route to the Gemini OpenAI-compatible API.
GEMINI_MODEL_PREFIXES = ("gemini/", "google/gemini-")
GLHF_BASE_URL = "https://glhf.chat/api/openai/v1"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
OLLAMA_API_KEY = "ollama"

VLLM_API_KEY = os.environ.get("VLLM_API_KEY", DUMMY_API_KEY)
LLAMACPP_API_KEY = os.environ.get("LLAMA_API_KEY", DUMMY_API_KEY)


openai_chat_model_pref_list = [
    OpenAIChatModel.GPT4o,
    OpenAIChatModel.GPT4_1_NANO,
    OpenAIChatModel.GPT4_1_MINI,
    OpenAIChatModel.GPT4_1,
    OpenAIChatModel.GPT4o_MINI,
    OpenAIChatModel.O1_MINI,
    OpenAIChatModel.O3_MINI,
    OpenAIChatModel.O1,
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


class OpenAICallParams(BaseModel):
    """
    Various params that can be sent to an OpenAI API chat-completion call.
    When specified, any param here overrides the one with same name in the
    OpenAIGPTConfig.
    See OpenAI API Reference for details on the params:
    https://platform.openai.com/docs/api-reference/chat
    """

    max_tokens: int | None = None
    temperature: float | None = None
    frequency_penalty: float | None = None  # between -2 and 2
    presence_penalty: float | None = None  # between -2 and 2
    response_format: Dict[str, str] | None = None
    logit_bias: Dict[int, float] | None = None  # token_id -> bias
    logprobs: bool | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None  # or "low" or "high" or "medium"
    top_logprobs: int | None = None  # if int, requires logprobs=True
    n: int | None = None  # how many completions to generate (n > 1 is NOT handled now)
    stop: str | List[str] | None = None  # (list of) stop sequence(s)
    seed: int | None = None
    user: str | None = None  # user id for tracking
    extra_body: Dict[str, Any] | None = None  # additional params for API request body

    def to_dict_exclude_none(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class LiteLLMProxyConfig(BaseSettings):
    """Configuration for LiteLLM proxy connection."""

    api_key: str = ""  # read from env var LITELLM_API_KEY if set
    api_base: str = ""  # read from env var LITELLM_API_BASE if set

    model_config = SettingsConfigDict(env_prefix="LITELLM_")


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
    # Callable returning a fresh API key/bearer token, for endpoints that
    # authenticate with short-lived rotating credentials (e.g. Vertex AI
    # OAuth tokens, Azure Entra ID). Resolved per-request by the OpenAI
    # client, and excluded from the client-cache key (the cache is keyed on
    # the provider's identity), so rotating tokens neither go stale nor grow
    # the cache. Takes precedence over `api_key`. Only supported for models
    # served via an OpenAI-compatible endpoint (not Groq/Cerebras/litellm).
    # See docs/notes/rotating-api-keys.md.
    api_key_provider: Optional[Callable[[], str]] = None
    organization: str = ""
    api_base: str | None = None  # used for local or other non-OpenAI models
    litellm: bool = False  # use litellm api?
    litellm_proxy: LiteLLMProxyConfig = LiteLLMProxyConfig()
    ollama: bool = False  # use ollama's OpenAI-compatible endpoint?
    min_output_tokens: int = 1
    use_chat_for_completion: bool = True  # do not change this, for OpenAI models!
    timeout: int = 20
    temperature: float = 0.2
    seed: int | None = 42
    params: OpenAICallParams | None = None
    # Pro-active rate limiting, paced from the provider's own rate-limit
    # headers. OFF by default; with `enabled=False` the request path is
    # unchanged. See docs/notes/rate-limiting.md.
    rate_limit: RateLimitConfig = RateLimitConfig()
    use_cached_client: bool = (
        True  # Whether to reuse cached clients (prevents resource exhaustion)
    )
    # these can be any model name that is served at an OpenAI-compatible API end point
    chat_model: str = default_openai_chat_model
    chat_model_orig: Optional[str] = None
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
    portkey_params: PortkeyParams = PortkeyParams()
    headers: Dict[str, str] = {}
    http_client_factory: Optional[Callable[[], Any]] = (
        None  # Factory: returns Client or (Client, AsyncClient)
    )
    http_verify_ssl: bool = True  # Simple flag for SSL verification
    http_client_config: Optional[Dict[str, Any]] = None  # Config dict for httpx.Client
    _api_base_was_supplied: bool = False

    def __init__(self, **kwargs) -> None:  # type: ignore
        api_base_was_supplied = "api_base" in kwargs
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
        env_prefix = self.model_config.get("env_prefix")
        env_api_base_name = f"{env_prefix}API_BASE"
        env_api_base = os.getenv(env_api_base_name)
        if not self.model_config.get("case_sensitive"):
            case_insensitive_env = {
                name.lower(): value for name, value in os.environ.items()
            }
            env_api_base = case_insensitive_env.get(env_api_base_name.lower())
        api_base_was_supplied = api_base_was_supplied or (
            self.api_base is not None
            and (env_prefix != "OPENAI_" or self.api_base != env_api_base)
        )
        self._api_base_was_supplied = api_base_was_supplied

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    def __setattr__(self, name: str, value: Any) -> None:
        """Track API bases assigned after config construction."""
        super().__setattr__(name, value)
        if name == "api_base":
            self._api_base_was_supplied = True

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> "OpenAIGPTConfig":
        """
        Copy config while preserving nested model instances and subclasses.

        Important: Avoid reconstructing via `model_dump` as that coerces nested
        models to their annotated base types (dropping subclass-only fields).
        Instead, defer to Pydantic's native `model_copy`, which keeps nested
        `BaseModel` instances (and their concrete subclasses) intact.

        An `api_base` supplied through `update` is caller configuration, so the
        copy must retain that provenance for provider-specific routing.
        """
        # Delegate to BaseSettings/BaseModel implementation to preserve types
        copied = super().model_copy(update=update, deep=deep)
        if update is not None and "api_base" in update:
            copied._api_base_was_supplied = True
        return copied  # type: ignore[return-value]

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
        litellm.modify_params = True
        self.seed = None  # some local mdls don't support seed

        if self.api_key == DUMMY_API_KEY:
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

        DynamicConfig.model_config = SettingsConfigDict(env_prefix=prefix.upper() + "_")
        return DynamicConfig


VERTEXAI_PREFIX = "vertexai/"
DEFAULT_VERTEXAI_LOCATION = "us-central1"
# GCP project ids and regions are lowercase alphanumerics, hyphen-separated.
# Anything else -- `/`, `@`, `:`, `.`, `%`, whitespace -- could change the
# authority or the path of the endpoint URL these values are spliced into.
# Matched with `fullmatch`: `re.match(...$)` would accept a trailing newline,
# so "us-central1\n" would pass and go straight into the URL.
_VERTEXAI_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def _validate_vertexai_id(field: str, value: str) -> str:
    """Validate a GCP project id or region before it is spliced into a URL.

    Applied at every entry point -- the `VertexAIConfig` fields AND the
    `GOOGLE_CLOUD_*`/`GCP_*` environment fallbacks, which pydantic never sees.
    An unvalidated value here would let whoever controls it redirect the
    request, and the ADC bearer token with it, to a host of their choosing.

    Args:
        field: name of the value being validated, for the error message.
        value: the candidate project id or region.

    Returns:
        `value`, unchanged, if it is safe to interpolate into the endpoint URL.

    Raises:
        ValueError: if `value` is empty, does not begin and end with a
            lowercase letter or digit, or contains anything but lowercase
            letters, digits and hyphens; or if it is the non-regional
            location `global`.
    """
    if not _VERTEXAI_ID_RE.fullmatch(value):
        raise ValueError(
            f"Invalid Vertex AI {field} {value!r}: must contain only lowercase "
            "letters, digits and hyphens, and begin and end with a letter or "
            "digit."
        )
    if field == "location" and value == "global":
        raise ValueError(
            "Vertex AI location 'global' has no regional OpenAI-compatible "
            f"endpoint; use a region such as '{DEFAULT_VERTEXAI_LOCATION}'."
        )
    return value


class VertexAIConfig(OpenAIGPTConfig):
    """
    Config for Google Vertex AI models served at its OpenAI-compatible endpoint,
    i.e. `chat_model="vertexai/<publisher>/<model>"`.

    This is a separate config class, rather than extra fields on
    `OpenAIGPTConfig`, for one reason: `OpenAIGPTConfig` is a pydantic
    `BaseSettings` with `env_prefix="OPENAI_"`, so a config built in a process
    that has `OPENAI_API_KEY`, `OPENAI_HEADERS`, `OPENAI_ORGANIZATION` or
    `OPENAI_API_BASE` set inherits every one of them -- and would then send
    them to Google. Overriding `env_prefix` to `VERTEXAI_` means those
    variables are not an env source for this class at all, so there is nothing
    to inherit: the fix is structural rather than a field-by-field scrub.

    Credentials, in precedence order:

    - `api_key_provider`, called per request (so a rotating token never
      goes stale);
    - `api_key` (or `VERTEXAI_API_KEY`);
    - otherwise Google Application Default Credentials, installed as an
      `api_key_provider`, which requires `gcloud auth application-default
      login` or `GOOGLE_APPLICATION_CREDENTIALS`.

    See `docs/notes/gemini.md`.
    """

    type: str = "vertexai"
    # GCP project; falls back to GOOGLE_CLOUD_PROJECT, then GCP_PROJECT
    project_id: Optional[str] = None
    # GCP region; falls back to GOOGLE_CLOUD_LOCATION, then us-central1
    location: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="VERTEXAI_")

    @field_validator("project_id", "location")
    @classmethod
    def _validate_project_and_location(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        if v is None:
            return v
        return _validate_vertexai_id(info.field_name or "value", v)


# Fields that must NOT be carried over from an `OPENAI_`-prefixed config onto
# the `vertexai/` route: each one can be populated from an `OPENAI_*` env var
# AND changes where the request goes, what it carries, or how it is secured, so
# carrying it over would reintroduce exactly the leak this route exists to
# prevent. `OPENAI_HTTP_CLIENT_CONFIG` is the subtle one -- it is spread into
# `httpx.Client(**config)`, so it can attach headers to, or proxy, the Vertex
# request; `OPENAI_HTTP_VERIFY_SSL=false` would disable TLS verification on the
# connection to Google; and `OPENAI_CHAT_MODEL_ORIG` decides which provider
# branch claims the request, so it could route a `vertexai/` model to the
# public Gemini endpoint with no ADC credential. (For `chat_model_orig` this
# is defense in depth: what actually keeps the route correct is testing
# `is_vertexai` first in the provider chain, plus recomputing
# `self.chat_model_orig` after the conversion. Dropping it here just stops a
# route-deciding value from living on a config that claims to be clean.)
# `OPENAI_LITELLM=true` is the same hazard by a different door: `litellm` is
# consulted as `startswith("litellm/") or config.litellm`, so it diverts the
# route to the litellm adapter -- which also sidesteps the guard that forbids
# `api_key_provider` there, leaving ADC silently unused.
#
# `api_key_provider` and `http_client_factory` are deliberately absent: both
# are callables, so they can only ever have been set explicitly in code, never
# by the environment.
#
# Anything added to `OpenAIGPTConfig` must be classified as dropped here or as
# safe to carry; `test_every_openai_config_field_is_classified` enforces that.
_VERTEXAI_DROPPED_FIELDS = frozenset(
    {
        "api_key",
        "headers",
        "organization",
        "api_base",
        "type",
        "http_client_config",
        "http_verify_ssl",
        "chat_model_orig",
        "litellm",
    }
)

# Dropping a non-default value in one of these is worth saying out loud: the
# caller may have set it in code (a private api_base, or a corporate proxy in
# http_client_config, are the cases that bite), and on this route there is no
# way to tell that apart from an `OPENAI_*` env value, so it has to go either
# way. Every dropped field that a caller plausibly sets in code belongs here --
# `api_key` most of all, since dropping a token the caller passed in code and
# then falling back to ADC is otherwise indistinguishable from a bug. Only
# `type` and `chat_model_orig` are excluded, as internal bookkeeping. Only the
# field NAME is ever logged, never its value.
_VERTEXAI_NOTIFY_IF_SET = (
    "api_key",
    "api_base",
    "headers",
    "organization",
    "http_client_config",
    "http_verify_ssl",
    "litellm",
)


def _as_vertexai_config(config: OpenAIGPTConfig) -> VertexAIConfig:
    """Rebuild `config` as a clean `VertexAIConfig`.

    Every field the caller set is carried over -- `temperature`,
    `max_output_tokens`, `use_cached_client`, `http_client_factory` and the
    rest -- except those in `_VERTEXAI_DROPPED_FIELDS`, which are left at
    `VertexAIConfig`'s defaults.

    Fields declared only on a custom `OpenAIGPTConfig` subclass cannot be
    carried over, since `VertexAIConfig` has no such fields; those are warned
    about rather than dropped silently. Subclass `VertexAIConfig` instead if
    you need them on a `vertexai/` route.

    `params` is carried but with two sub-fields cleared, because
    `OPENAI_PARAMS` sets the whole nested model and both of these travel in
    the request body: `extra_body`, an arbitrary dict (and provider-specific
    by definition, so carrying one from an OpenAI config to Google would be
    wrong even with no environment involved), and `user`, an end-user
    identifier the provider logs -- the same kind of identifier as
    `organization`, which is dropped outright. The rest of `params`
    (`top_p`, `stop`, ...) is generation behavior and is kept.
    """
    source_fields = type(config).model_fields
    carried = {
        name: getattr(config, name)
        for name in source_fields
        if name in VertexAIConfig.model_fields and name not in _VERTEXAI_DROPPED_FIELDS
    }
    unsupported = sorted(set(source_fields) - set(VertexAIConfig.model_fields))
    if unsupported:
        logging.warning(
            "vertexai/ route: dropping config fields %s, which exist only on "
            "%s. Subclass VertexAIConfig to keep them.",
            ", ".join(unsupported),
            type(config).__name__,
        )
    deliberate = [
        name
        for name in _VERTEXAI_NOTIFY_IF_SET
        if name in source_fields
        and getattr(config, name) != source_fields[name].get_default()
    ]
    if deliberate:
        logging.warning(
            "vertexai/ route: not carrying over config fields %s -- on this "
            "route a value there cannot be told apart from an "
            "OPENAI_-prefixed environment value. If it was meant for Vertex "
            "AI, set it via VERTEXAI_* or on a VertexAIConfig you construct "
            "directly.",
            ", ".join(deliberate),
        )
    params = carried.get("params")
    if params is not None:
        body_identifying = [
            name for name in ("extra_body", "user") if getattr(params, name) is not None
        ]
        if body_identifying:
            logging.warning(
                "vertexai/ route: clearing params.%s, which OPENAI_PARAMS can "
                "set and which would be sent in the request body to Google. "
                "Set it on a VertexAIConfig you construct directly if it is "
                "meant for Vertex AI.",
                ", params.".join(body_identifying),
            )
            carried["params"] = params.model_copy(
                update={name: None for name in body_identifying}
            )
    return VertexAIConfig(**carried)


# Environment variables the `openai` SDK itself reads inside `OpenAI.__init__`,
# for arguments langroid does not pass. Swapping `VertexAIConfig`'s env prefix
# cannot close these: the SDK reads them from the environment directly, not
# from our config, so they bypass the clean-config mechanism entirely.
#
#   OPENAI_PROJECT_ID     -> sent as an `OpenAI-Project` header on every request
#   OPENAI_CUSTOM_HEADERS -> parsed into ARBITRARY headers and merged into
#                            default_headers: the OPENAI_HEADERS hazard again,
#                            one layer down, and there is no argument to
#                            override it with
#   OPENAI_ADMIN_KEY      -> stored on the client
#   OPENAI_WEBHOOK_SECRET -> stored on the client
#
# `OPENAI_API_KEY`, `OPENAI_ORG_ID` and `OPENAI_BASE_URL` are already closed,
# because langroid always passes `api_key`, `organization` and `base_url`
# explicitly, and the SDK only consults the environment when the argument is
# absent.
_OPENAI_SDK_ENV_VARS = (
    "OPENAI_PROJECT_ID",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_ADMIN_KEY",
    "OPENAI_WEBHOOK_SECRET",
)


@contextmanager
def _without_openai_sdk_env() -> Iterator[None]:
    """Hide the SDK's own `OPENAI_*` channels while a Vertex client is built.

    Removing the variables is the only way to close them: they have no
    constructor argument to override (`OPENAI_CUSTOM_HEADERS`) or one langroid
    does not pass (`OPENAI_PROJECT_ID`). Enumerating them in a deny-list of
    config fields cannot work, since they are never config fields.

    Caveat, stated rather than hidden: `os.environ` is process-global, so this
    is not thread-safe against another thread constructing a client for a
    genuinely-OpenAI route in the same instant. The window is a single client
    construction; the alternative is sending those values to Google, so the
    trade is deliberate.
    """
    saved = {k: os.environ[k] for k in _OPENAI_SDK_ENV_VARS if k in os.environ}
    for k in saved:
        del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(saved)


_adc_lock = threading.Lock()
_adc_credentials: Any = None


def _adc_access_token() -> str:
    """Return a valid Google ADC access token, refreshing it only when stale.

    Installed as an `api_key_provider`, so it is called on every request. The
    credentials object is therefore created once and cached at module level:
    `google-auth` tracks the token's expiry on it, so this refreshes roughly
    hourly instead of making a token round trip per LLM call. Being a plain
    module-level function, its identity is stable, which keeps the OpenAI
    client cache (keyed on the provider's identity) from churning.
    """
    global _adc_credentials
    try:
        import google.auth
        import google.auth.transport.requests
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError as e:
        # No extras name: google-auth has no langroid extra of its own, it
        # arrives transitively with google-api-python-client.
        raise LangroidImportError("google-auth", error=str(e)) from e
    with _adc_lock:
        if _adc_credentials is None:
            try:
                _adc_credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except DefaultCredentialsError as e:
                raise ValueError(
                    "Google Application Default Credentials not found, which a "
                    "vertexai/ model needs: run `gcloud auth "
                    "application-default login`, or point "
                    "GOOGLE_APPLICATION_CREDENTIALS at a service-account key, "
                    "or set VERTEXAI_API_KEY to supply a token directly."
                ) from e
        if not _adc_credentials.valid:
            _adc_credentials.refresh(google.auth.transport.requests.Request())
        token = _adc_credentials.token
    if not token:
        raise ValueError("Google ADC returned an empty access token.")
    return str(token)


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
        # copy the config to avoid modifying the original; deep to decouple
        # nested models while preserving their concrete subclasses
        # The api_key_provider must NOT be deep-copied: the client cache is
        # keyed on its identity (so a copy would defeat cache sharing), and it
        # may hold non-copyable state (e.g. a threading.Lock). Clear it in a
        # shallow copy first, then restore the original callable afterwards.
        api_key_provider = config.api_key_provider
        if api_key_provider is not None:
            config = config.model_copy(update={"api_key_provider": None})
        config = config.model_copy(deep=True)
        if api_key_provider is not None:
            config.api_key_provider = api_key_provider
        super().__init__(config)
        self.config: OpenAIGPTConfig = config
        # save original model name such as `provider/model` before
        # we strip out the `provider` - we retain the original in
        # case some params are specific to a provider.
        self.chat_model_orig = self.config.chat_model_orig or self.config.chat_model

        # Run the first time the model is used
        self.run_on_first_use = cache(self.config.run_on_first_use)

        # global override of chat_model,
        # to allow quick testing with other models
        if settings.chat_model != "":
            self.config.chat_model = settings.chat_model
            self.chat_model_orig = settings.chat_model
            self.config.completion_model = settings.chat_model

        # Whether the caller asked for a vertexai/ route, recorded BEFORE the
        # `model//formatter` split below can eat the prefix.
        asked_for_vertexai = self.config.chat_model.startswith(VERTEXAI_PREFIX)

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

        if asked_for_vertexai and not self.config.chat_model.startswith(
            VERTEXAI_PREFIX
        ):
            # The `//formatter` suffix consumed the route itself, e.g.
            # "vertexai//hf" -> the bare model "vertexai". Left alone this
            # falls through to the generic branch and quietly talks to
            # api.openai.com, which is never what someone typing `vertexai`
            # meant. (On a VertexAIConfig it would also take a Google
            # credential there -- the refusal below catches that case too, but
            # this one fires for a plain OpenAIGPTConfig as well.)
            raise ValueError(
                f"chat_model {self.chat_model_orig!r} is not a usable "
                "vertexai/ route: the //formatter suffix left no model behind. "
                "Write vertexai/<publisher>/<model>//<formatter>, e.g. "
                "vertexai/google/gemini-2.5-flash//hf."
            )

        # A vertexai/ route must not inherit OPENAI_-prefixed settings, so
        # rebuild the config as a clean VertexAIConfig.
        #
        # Placed here deliberately, and the position is load-bearing twice
        # over: AFTER the settings.chat_model override above, so the direct
        # route and the override path get the same clean config; and AFTER the
        # `model//formatter` split just above, so both branches below see the
        # final chat_model. Before the split, "vertexai//hf" still carried the
        # prefix and slipped past both branches -- then the split turned it
        # into the bare model "vertexai", which routes to api.openai.com,
        # taking a VERTEXAI_API_KEY with it.
        if self.config.chat_model.startswith(VERTEXAI_PREFIX) and not isinstance(
            self.config, VertexAIConfig
        ):
            self.config = _as_vertexai_config(self.config)
            # The conversion drops chat_model_orig (OPENAI_CHAT_MODEL_ORIG
            # could otherwise decide which provider branch claims this
            # request), so recompute it from the model actually in effect.
            self.chat_model_orig = self.config.chat_model
        elif isinstance(self.config, VertexAIConfig) and not (
            self.config.chat_model.startswith(VERTEXAI_PREFIX)
        ):
            # The converse of the above, and the same leak running backwards: a
            # VertexAIConfig carries a credential from VERTEXAI_API_KEY, and
            # without the vertexai/ prefix this would fall through to the
            # generic branch -- api_base None, i.e. api.openai.com -- and send
            # that credential there. Reachable via the settings.chat_model
            # override alone (`-m gpt-4o`), so refuse rather than guess.
            raise ValueError(
                f"chat_model {self.config.chat_model!r} is not a vertexai/ "
                "route, but the config is a VertexAIConfig, whose credential "
                "is meant for Google. Use an OpenAIGPTConfig for non-Vertex "
                "models. (If this came from a global chat_model override such "
                "as `-m <model>`, that override cannot be applied to a "
                "VertexAIConfig. If you passed an existing llm.config back in, "
                "note that its vertexai/ prefix was already stripped.)"
            )

        if self.config.formatter is not None:
            self.config.hf_formatter = HFFormatter(
                HFPromptFormatterConfig(model_name=self.config.formatter)
            )

        self.supports_json_schema: bool = self.config.supports_json_schema or False
        self.supports_strict_tools: bool = self.config.supports_strict_tools or False

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DUMMY_API_KEY)
        # read from self.config, not the `config` argument: a vertexai/ route
        # replaced self.config above, and the original still holds the
        # OPENAI_-prefixed key we must not use.
        self.api_key = self.config.api_key

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
        self.is_minimax = self.is_minimax_model()
        self.is_glhf = self.config.chat_model.startswith("glhf/")
        self.is_openrouter = self.config.chat_model.startswith("openrouter/")
        self.is_langdb = self.config.chat_model.startswith("langdb/")
        self.is_portkey = self.config.chat_model.startswith("portkey/")
        self.is_vertexai = self.config.chat_model.startswith(VERTEXAI_PREFIX)
        self.is_litellm_proxy = self.config.chat_model.startswith("litellm-proxy/")

        if self.config.api_key_provider is not None and (
            self.is_groq or self.is_cerebras or self.config.litellm
        ):
            raise ValueError(
                "api_key_provider is only supported for models served via an "
                "OpenAI-compatible endpoint (using the OpenAI client); it "
                "cannot be used with Groq, Cerebras, or the litellm adapter."
            )

        if self.is_groq:
            # use groq-specific client
            self.config.chat_model = self.config.chat_model.replace("groq/", "")
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.getenv("GROQ_API_KEY", DUMMY_API_KEY)
            if self.config.use_cached_client:
                self.client = get_groq_client(api_key=self.api_key)
                self.async_client = get_async_groq_client(api_key=self.api_key)
            else:
                # Create new clients without caching
                self.client = Groq(api_key=self.api_key)
                self.async_client = AsyncGroq(api_key=self.api_key)
        elif self.is_cerebras:
            # use cerebras-specific client
            self.config.chat_model = self.config.chat_model.replace("cerebras/", "")
            if self.api_key == OPENAI_API_KEY:
                self.api_key = os.getenv("CEREBRAS_API_KEY", DUMMY_API_KEY)
            if self.config.use_cached_client:
                self.client = get_cerebras_client(api_key=self.api_key)
                # TODO there is not async client, so should we do anything here?
                self.async_client = get_async_cerebras_client(api_key=self.api_key)
            else:
                # Create new clients without caching
                self.client = Cerebras(api_key=self.api_key)
                self.async_client = AsyncCerebras(api_key=self.api_key)
        else:
            # in these cases, there's no specific client: OpenAI python client suffices
            #
            # vertexai/ is tested FIRST, deliberately: the other branches key
            # off `chat_model_orig`, and if one of them claimed a vertexai/
            # model the route would silently become that provider's -- a public
            # endpoint, with no ADC credential and no regional URL.
            if self.is_vertexai:
                # self.config was rebuilt as a VertexAIConfig above, so this
                # branch cannot see an OPENAI_-prefixed api_key/headers.
                vertex_config = cast(VertexAIConfig, self.config)
                # "vertexai/google/gemini-2.5-flash" -> "google/gemini-2.5-flash",
                # which is the model id the endpoint expects.
                vertex_model = vertex_config.chat_model[len(VERTEXAI_PREFIX) :]
                publisher, _, bare_model = vertex_model.partition("/")
                if not publisher or not bare_model:
                    raise ValueError(
                        f"Invalid vertexai/ model {vertex_config.chat_model!r}: "
                        "expected vertexai/<publisher>/<model>, e.g. "
                        "vertexai/google/gemini-2.5-flash."
                    )
                vertex_config.chat_model = vertex_model
                if vertex_config.completion_model.startswith(VERTEXAI_PREFIX):
                    vertex_config.completion_model = vertex_config.completion_model[
                        len(VERTEXAI_PREFIX) :
                    ]
                if vertex_config.api_base:
                    # An explicit api_base (VERTEXAI_API_BASE, or api_base on a
                    # directly-constructed VertexAIConfig) targets a private/PSC
                    # endpoint; honor it rather than overwriting it. The
                    # conversion never carries api_base over from an
                    # OPENAI_-prefixed config, so a value here is deliberate.
                    self.api_base = vertex_config.api_base
                else:
                    vertex_project = (
                        vertex_config.project_id
                        or os.getenv("GOOGLE_CLOUD_PROJECT")
                        or os.getenv("GCP_PROJECT")
                    )
                    if not vertex_project:
                        raise ValueError(
                            "A GCP project is required for a vertexai/ model: "
                            "set VertexAIConfig(project_id=...), or "
                            "VERTEXAI_PROJECT_ID, GOOGLE_CLOUD_PROJECT or "
                            "GCP_PROJECT in the environment."
                        )
                    vertex_location = (
                        vertex_config.location
                        or os.getenv("GOOGLE_CLOUD_LOCATION")
                        or DEFAULT_VERTEXAI_LOCATION
                    )
                    # Re-validate: the values may have come from
                    # GOOGLE_CLOUD_*/GCP_* above, which the field validators
                    # never see, and both are interpolated into the URL below.
                    vertex_config.project_id = _validate_vertexai_id(
                        "project_id", vertex_project
                    )
                    vertex_config.location = _validate_vertexai_id(
                        "location", vertex_location
                    )
                    self.api_base = (
                        f"https://{vertex_config.location}-aiplatform."
                        f"googleapis.com/v1beta1"
                        f"/projects/{vertex_config.project_id}"
                        f"/locations/{vertex_config.location}/endpoints/openapi"
                    )
                no_explicit_key = vertex_config.api_key in ("", DUMMY_API_KEY)
                if vertex_config.api_key_provider is None and no_explicit_key:
                    # No explicit Vertex credential: fall back to Google ADC,
                    # resolved per request so its ~1h lifetime is handled here
                    # rather than by the caller.
                    vertex_config.api_key_provider = _adc_access_token
            elif self.is_litellm_proxy:
                self.config.chat_model = self.config.chat_model.replace(
                    "litellm-proxy/", ""
                )
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = self.config.litellm_proxy.api_key or self.api_key
                self.api_base = self.config.litellm_proxy.api_base or self.api_base
            elif self.is_gemini:
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = os.getenv("GEMINI_API_KEY", DUMMY_API_KEY)
                # Prefer caller config, then Gemini env, then the default.
                gemini_api_base = os.getenv("GEMINI_API_BASE", "")
                explicit_api_base = (
                    self.config.api_base if self.config._api_base_was_supplied else None
                )
                self.api_base = explicit_api_base or gemini_api_base or GEMINI_BASE_URL
                if (
                    self.config.chat_model.startswith("gemini/")
                    or self.api_base == GEMINI_BASE_URL
                ):
                    self.config.chat_model = self.config.chat_model.split("/", 1)[1]
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
            elif self.is_minimax:
                self.config.chat_model = self.config.chat_model.replace("minimax/", "")
                # Honor caller-supplied base URL (e.g. regional endpoints,
                # proxies) instead of always forcing the default.
                openai_api_base = os.getenv("OPENAI_API_BASE")
                explicit_api_base = (
                    self.config.api_base
                    if self.config.api_base and self.config.api_base != openai_api_base
                    else None
                )
                self.api_base = explicit_api_base or MINIMAX_BASE_URL
                if self.api_key == OPENAI_API_KEY:
                    # Only overwrite with MINIMAX_API_KEY when it is actually
                    # set, so users who intentionally put their MiniMax key in
                    # OPENAI_API_KEY are not silently downgraded to a dummy key.
                    minimax_key = os.getenv("MINIMAX_API_KEY", "")
                    if minimax_key:
                        self.api_key = minimax_key
                # Recompute capabilities now that the prefix has been stripped
                # and self.info() can find the model in MODEL_INFO.
                self.supports_strict_tools = True
                self.supports_json_schema = self.info().has_structured_output
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
            elif self.is_portkey:
                # Parse the model string and extract provider/model
                provider, model = self.config.portkey_params.parse_model_string(
                    self.config.chat_model
                )
                self.config.chat_model = model
                if provider:
                    self.config.portkey_params.provider = provider

                # Set Portkey base URL
                self.api_base = self.config.portkey_params.base_url + "/v1"

                # Set API key - use provider's API key from env if available
                if self.api_key == OPENAI_API_KEY:
                    self.api_key = self.config.portkey_params.get_provider_api_key(
                        self.config.portkey_params.provider, DUMMY_API_KEY
                    )

                # Add Portkey-specific headers
                self.config.headers.update(self.config.portkey_params.get_headers())

            # Sanitize the API key: strip leading/trailing whitespace
            # (including stray newlines from .env files or CI secrets).
            self.api_key = self.api_key.strip()

            # Create http_client if needed - Priority order:
            # 1. http_client_factory (most flexibility, not cacheable)
            # 2. http_client_config (cacheable, moderate flexibility)
            # 3. http_verify_ssl=False (cacheable, simple SSL bypass)
            http_client = None
            async_http_client = None
            http_client_config_used = None

            if self.config.http_client_factory is not None:
                # Use the factory to create http_client (not cacheable)
                http_client = self.config.http_client_factory()
                if isinstance(http_client, (list, tuple)):
                    if len(http_client) != 2:
                        raise ValueError(
                            "http_client_factory must return either a single "
                            "httpx.Client or a tuple of "
                            "(httpx.Client, httpx.AsyncClient)"
                        )
                    http_client, async_http_client = http_client
                else:
                    # set async_http_client to None - so that it will
                    # be created later
                    async_http_client = None
            elif self.config.http_client_config is not None:
                # Use config dict (cacheable)
                http_client_config_used = self.config.http_client_config
            elif not self.config.http_verify_ssl:
                # Simple SSL bypass (cacheable)
                http_client_config_used = {"verify": False}
                logging.warning(
                    "SSL verification has been disabled. This is insecure and "
                    "should only be used in trusted environments (e.g., "
                    "corporate networks with self-signed certificates)."
                )

            # With an api_key_provider, hand the callable itself to the
            # OpenAI client, which resolves a fresh token on each request.
            openai_api_key: Union[str, Callable[[], str]] = (
                self.config.api_key_provider
                if self.config.api_key_provider is not None
                else self.api_key
            )

            # On a vertexai/ route, hide the SDK's own OPENAI_* channels
            # for the duration of client construction -- see
            # _without_openai_sdk_env. A nullcontext elsewhere, so no
            # other route changes behaviour.
            with _without_openai_sdk_env() if self.is_vertexai else nullcontext():
                if self.config.use_cached_client:
                    self.client = get_openai_client(
                        api_key=openai_api_key,
                        base_url=self.api_base,
                        organization=self.config.organization,
                        timeout=Timeout(self.config.timeout),
                        default_headers=self.config.headers,
                        http_client=http_client,
                        http_client_config=http_client_config_used,
                        sdk_env_scrubbed=self.is_vertexai,
                    )
                    self.async_client = get_async_openai_client(
                        api_key=openai_api_key,
                        base_url=self.api_base,
                        organization=self.config.organization,
                        timeout=Timeout(self.config.timeout),
                        default_headers=self.config.headers,
                        http_client=async_http_client,
                        http_client_config=http_client_config_used,
                        sdk_env_scrubbed=self.is_vertexai,
                    )
                else:
                    # Create new clients without caching
                    client_kwargs: Dict[str, Any] = dict(
                        api_key=openai_api_key,
                        base_url=self.api_base,
                        organization=self.config.organization,
                        timeout=Timeout(self.config.timeout),
                        default_headers=self.config.headers,
                    )
                    if http_client is not None:
                        client_kwargs["http_client"] = http_client
                    elif http_client_config_used is not None:
                        # Create http_client from config for non-cached scenario
                        try:
                            httpx = import_httpx()
                        except ImportError:
                            raise ValueError(missing_httpx_message())
                        client_kwargs["http_client"] = httpx.Client(
                            **http_client_config_used
                        )
                    self.client = OpenAI(**client_kwargs)

                    async_client_kwargs: Dict[str, Any] = dict(
                        api_key=(
                            # AsyncOpenAI awaits its api_key callable
                            wrap_api_key_provider_async(openai_api_key)
                            if callable(openai_api_key)
                            else openai_api_key
                        ),
                        base_url=self.api_base,
                        organization=self.config.organization,
                        timeout=Timeout(self.config.timeout),
                        default_headers=self.config.headers,
                    )
                    if async_http_client is not None:
                        async_client_kwargs["http_client"] = async_http_client
                    elif http_client_config_used is not None:
                        # Create async http_client from config for non-cached scenario
                        try:
                            httpx = import_httpx()
                        except ImportError:
                            raise ValueError(missing_httpx_message())
                        async_client_kwargs["http_client"] = httpx.AsyncClient(
                            **http_client_config_used
                        )
                    self.async_client = AsyncOpenAI(**async_client_kwargs)

        self.cache: CacheDB | None = None
        use_cache = self.config.cache_config is not None
        if "redis" in settings.cache_type and use_cache:
            # read and write self.config, not the `config` argument: a
            # vertexai/ route replaced self.config above, and mutating the
            # original would leave self.config.cache_config disagreeing with
            # the cache actually built here.
            if self.config.cache_config is None or not isinstance(
                self.config.cache_config,
                RedisCacheConfig,
            ):
                # switch to fresh redis config if needed
                self.config.cache_config = RedisCacheConfig(
                    fake="fake" in settings.cache_type
                )
            if "fake" in settings.cache_type:
                # force use of fake redis if global cache_type is "fakeredis"
                self.config.cache_config.fake = True
            self.cache = RedisCache(self.config.cache_config)
        elif settings.cache_type != "none" and use_cache:
            raise ValueError(
                f"Invalid cache type {settings.cache_type}. "
                "Valid types are redis, fakeredis, none"
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
        return self.chat_model_orig.startswith(GEMINI_MODEL_PREFIXES)

    def is_deepseek_model(self) -> bool:
        deepseek_models = [e.value for e in DeepSeekModel]
        return (
            self.chat_model_orig in deepseek_models
            or self.chat_model_orig.startswith("deepseek/")
        )

    def is_minimax_model(self) -> bool:
        """Are we using the MiniMax OpenAI-compatible API?"""
        minimax_models = [e.value for e in MiniMaxModel]
        return (
            self.chat_model_orig in minimax_models
            or self.chat_model_orig.startswith("minimax/")
        )

    def unsupported_params(self) -> List[str]:
        """
        List of params that are not supported by the current model
        """
        unsupported = set(self.info().unsupported_params)
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

    def chat_cost(self) -> Tuple[float, float, float]:
        """
        (Prompt, Cached, Generation) cost per 1000 tokens, for chat-completion
        models/endpoints.
        Get it from the dict, otherwise fail-over to general method
        """
        info = self.info()
        cached_cost_per_million = info.cached_cost_per_million
        if not cached_cost_per_million:
            cached_cost_per_million = info.input_cost_per_million
        return (
            info.input_cost_per_million / 1000,
            cached_cost_per_million / 1000,
            info.output_cost_per_million / 1000,
        )

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

    @staticmethod
    def _split_inline_reasoning(
        event_text: str,
        event_reasoning: str,
        in_reasoning: bool,
        thought_delimiters: Tuple[str, str],
    ) -> Tuple[str, str, bool]:
        """Separate inline reasoning from text tokens in a streaming chunk.

        When models embed thinking inside content (e.g. <think>...</think>)
        rather than using a separate reasoning field, this splits the chunk
        into text-only and reasoning-only portions for proper streamer routing.

        Returns (text_tokens, reasoning_tokens, in_reasoning).
        """
        text_tokens = event_text
        reasoning_tokens = event_reasoning

        if not event_text or event_reasoning:
            return text_tokens, reasoning_tokens, in_reasoning

        start, end = thought_delimiters
        remaining = event_text

        if in_reasoning:
            text_tokens = ""
        elif start in event_text:
            before, _, after = event_text.partition(start)
            text_tokens = before
            remaining = after
            in_reasoning = True

        if in_reasoning:
            if end in remaining:
                before, _, after = remaining.partition(end)
                text_tokens += after
                reasoning_tokens = before
                in_reasoning = False
            else:
                reasoning_tokens = remaining

        return text_tokens, reasoning_tokens, in_reasoning

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
        in_reasoning: bool = False,
    ) -> Tuple[bool, bool, str, str, bool, Dict[str, int]]:
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
                in_reasoning,
                usage,
            )
        event_args = ""
        event_fn_name = ""
        event_tool_deltas: Optional[List[Dict[str, Any]]] = None
        silent = settings.quiet
        # The first two events in the stream of Azure OpenAI is useless.
        # In the 1st: choices list is empty, in the 2nd: the dict delta has null content
        if chat:
            delta = choices[0].get("delta", {}) or {}
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

        event_text_tokens, event_reasoning_tokens, in_reasoning = (
            self._split_inline_reasoning(
                event_text,
                event_reasoning,
                in_reasoning,
                self.config.thought_delimiters,
            )
        )

        if event_text:
            completion += event_text
        if event_text_tokens:
            if not silent:
                sys.stdout.write(Colors().GREEN + event_text_tokens)
                sys.stdout.flush()
            self.config.streamer(event_text_tokens, StreamEventType.TEXT)

        if event_reasoning:
            reasoning += event_reasoning
        if event_reasoning_tokens:
            if not silent:
                sys.stdout.write(Colors().GREEN_DIM + event_reasoning_tokens)
                sys.stdout.flush()
            self.config.streamer(event_reasoning_tokens, StreamEventType.REASONING)

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
            in_reasoning,
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
        in_reasoning: bool = False,
    ) -> Tuple[bool, bool, str, str, bool, Dict[str, int]]:
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
                in_reasoning,
                usage,
            )
        event_args = ""
        event_fn_name = ""
        event_tool_deltas: Optional[List[Dict[str, Any]]] = None
        silent = self.config.async_stream_quiet or settings.quiet
        # The first two events in the stream of Azure OpenAI is useless.
        # In the 1st: choices list is empty, in the 2nd: the dict delta has null content
        if chat:
            delta = choices[0].get("delta", {}) or {}
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

        event_text_tokens, event_reasoning_tokens, in_reasoning = (
            self._split_inline_reasoning(
                event_text,
                event_reasoning,
                in_reasoning,
                self.config.thought_delimiters,
            )
        )

        if event_text:
            completion += event_text
        if event_text_tokens:
            if not silent:
                sys.stdout.write(Colors().GREEN + event_text_tokens)
                sys.stdout.flush()
            await self.config.streamer_async(event_text_tokens, StreamEventType.TEXT)

        if event_reasoning:
            reasoning += event_reasoning
        if event_reasoning_tokens:
            if not silent:
                sys.stdout.write(Colors().GREEN_DIM + event_reasoning_tokens)
                sys.stdout.flush()
            await self.config.streamer_async(
                event_reasoning_tokens, StreamEventType.REASONING
            )

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
            in_reasoning,
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
        in_reasoning: bool = False  # Track if we're inside reasoning delimiters
        content_present = False
        try:
            for event in response:
                event_dict = event if isinstance(event, dict) else event.model_dump()
                choices = event_dict.get("choices") or []
                if chat and choices:
                    delta = choices[0].get("delta") or {}
                    content_present = content_present or (
                        "content" in delta and delta["content"] is not None
                    )
                (
                    is_break,
                    has_function,
                    function_name,
                    function_args,
                    completion,
                    reasoning,
                    in_reasoning,
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
                    in_reasoning=in_reasoning,
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
            content_present=content_present,
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
        in_reasoning: bool = False  # Track if we're inside reasoning delimiters
        content_present = False
        try:
            async for event in response:
                event_dict = event if isinstance(event, dict) else event.model_dump()
                choices = event_dict.get("choices") or []
                if chat and choices:
                    delta = choices[0].get("delta") or {}
                    content_present = content_present or (
                        "content" in delta and delta["content"] is not None
                    )
                (
                    is_break,
                    has_function,
                    function_name,
                    function_args,
                    completion,
                    reasoning,
                    in_reasoning,
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
                    in_reasoning=in_reasoning,
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
            content_present=content_present,
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
                "extra_content": None,
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

            if tool_delta.get("extra_content") is not None:
                idx2tool_dict[tool_delta["index"]]["extra_content"] = tool_delta[
                    "extra_content"
                ]

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
                extra_content=tool_dict.get("extra_content"),
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
        content_present: bool = False,
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
            content_present: whether a stream delta explicitly contained content
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
            if failed_content:
                completion += ("\n" if completion else "") + failed_content
                content_present = True
            has_valid_call = len(tool_dicts) > 0 or has_function
            response_content = (
                None
                if has_valid_call and not content_present and completion == ""
                else completion
            )
            msg: Dict[str, Any] = dict(
                message=dict(
                    content=response_content,
                    reasoning_content=reasoning,
                ),
            )
            if len(tool_dicts) > 0:
                msg["message"]["tool_calls"] = tool_dicts

            if has_function:
                function_call = LLMFunctionCall(name=function_name)
                function_call_dict = function_call.model_dump()
                if function_args == "":
                    function_call.arguments = None
                else:
                    function_call.arguments = args
                    function_call_dict.update({"arguments": function_args.strip()})
                msg["message"]["function_call"] = function_call_dict
        else:
            # non-chat mode has no function_call
            msg = dict(text=completion)
            response_content = completion
            # TODO: Ignoring reasoning content for non-chat models

        # create an OpenAIResponse object so we can cache it as if it were
        # a non-streaming response
        openai_response = OpenAIResponse(
            choices=[msg],
            usage=dict(total_tokens=0),
        )
        # Track whether we extracted inline thought tags from the text.
        # Only set message_with_reasoning when get_reasoning_final()
        # actually finds and extracts inline tags (e.g. <think>...</think>).
        # When reasoning is already provided via a separate API field
        # (e.g. reasoning_content), the message text doesn't contain
        # thought signatures, so there's nothing extra to preserve.
        message_with_reasoning = None
        message: str | None
        if reasoning == "" and response_content is not None:
            # some LLM APIs may not return a separate reasoning field,
            # and the reasoning may be included in the message content
            # within delimiters like <think> ... </think>
            reasoning, message = self.get_reasoning_final(response_content)
            if reasoning:
                # Inline tags were found and extracted; preserve the
                # original text so it can be restored in message history.
                message_with_reasoning = response_content
        else:
            message = response_content

        prompt_tokens = usage.get("prompt_tokens", 0)
        prompt_tokens_details: Any = usage.get("prompt_tokens_details", {})
        cached_tokens = (
            prompt_tokens_details.get("cached_tokens", 0)
            if isinstance(prompt_tokens_details, dict)
            else 0
        )
        completion_tokens = usage.get("completion_tokens", 0)

        return (
            LLMResponse(
                message=message,
                reasoning=reasoning,
                message_with_reasoning=message_with_reasoning,
                cached=False,
                # don't allow empty list [] here
                oai_tool_calls=tool_calls or None if len(tool_deltas) > 0 else None,
                function_call=function_call if has_function else None,
                usage=LLMTokenUsage(
                    prompt_tokens=prompt_tokens or 0,
                    cached_tokens=cached_tokens or 0,
                    completion_tokens=completion_tokens or 0,
                    cost=self._cost_chat_model(
                        prompt_tokens or 0,
                        cached_tokens or 0,
                        completion_tokens or 0,
                    ),
                ),
            ),
            openai_response.model_dump(),
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

    def _cost_chat_model(self, prompt: int, cached: int, completion: int) -> float:
        price = self.chat_cost()
        return (
            price[0] * (prompt - cached) + price[1] * cached + price[2] * completion
        ) / 1000

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
        cached_tokens = 0
        completion_tokens = 0

        usage = response.get("usage")
        if not cached and not self.get_stream() and usage is not None:
            prompt_tokens = usage.get("prompt_tokens") or 0
            prompt_tokens_details = usage.get("prompt_tokens_details", {}) or {}
            cached_tokens = prompt_tokens_details.get("cached_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            cost = self._cost_chat_model(
                prompt_tokens or 0,
                cached_tokens or 0,
                completion_tokens or 0,
            )

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        self.run_on_first_use()

        try:
            return self._generate(prompt, max_tokens)
        except openai.APIStatusError as e:
            # Catch HTTP-level API errors (400, 401, 403, 404, 422, 429, 5xx)
            # without traceback — these originate server-side and a local
            # stack trace adds no diagnostic value.
            # Note: APIConnectionError/APITimeoutError are intentionally NOT
            # caught here so they fall through to the generic handler below,
            # where the full traceback aids in diagnosing local network issues.
            logging.error(f"API error in OpenAIGPT.generate: {e}")
            raise
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.generate: "))
            raise

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

                    if self.api_key != DUMMY_API_KEY:
                        kwargs["api_key"] = self.api_key
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
            response = response.model_dump()
        if "message" in response["choices"][0]:
            msg = (response["choices"][0]["message"]["content"] or "").strip()
        else:
            msg = (response["choices"][0]["text"] or "").strip()
        return LLMResponse(message=msg, cached=cached)

    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        self.run_on_first_use()

        try:
            return await self._agenerate(prompt, max_tokens)
        except openai.APIStatusError as e:
            # Catch HTTP-level API errors (see comment in generate() above).
            logging.error(f"API error in OpenAIGPT.agenerate: {e}")
            raise
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.agenerate: "))
            raise

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

                    if self.api_key != DUMMY_API_KEY:
                        kwargs["api_key"] = self.api_key

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
            response = response.model_dump()
        if "message" in response["choices"][0]:
            msg = (response["choices"][0]["message"]["content"] or "").strip()
        else:
            msg = (response["choices"][0]["text"] or "").strip()
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
        except openai.APIStatusError as e:
            # Catch HTTP-level API errors (see comment in generate() above).
            logging.error(f"API error in OpenAIGPT.chat: {e}")
            raise
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.chat: "))
            raise

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
        except openai.APIStatusError as e:
            # Catch HTTP-level API errors (see comment in generate() above).
            logging.error(f"API error in OpenAIGPT.achat: {e}")
            raise
        except Exception as e:
            # log and re-raise exception
            logging.error(friendly_error(e, "Error in OpenAIGPT.achat: "))
            raise

    def _rate_limiter(self) -> Optional[RateLimiter]:
        """The shared rate limiter for this model, or None when disabled.

        Returns None unless `config.rate_limit.enabled` is set, so that the
        request path is untouched by default.
        """
        cfg: Optional[RateLimitConfig] = getattr(self.config, "rate_limit", None)
        if cfg is None or not cfg.enabled:
            return None
        key = cfg.share_key or (
            f"{self.config.api_base or 'default'}::{self.config.chat_model}"
        )
        return get_rate_limiter(key, cfg)

    @staticmethod
    def _response_tokens(result: Any) -> Optional[int]:
        """Total tokens billed for a chat-completion response, if reported."""
        usage = getattr(result, "usage", None)
        total = getattr(usage, "total_tokens", None)
        return total if isinstance(total, int) else None

    def _chat_completions_with_backoff_body(self, **kwargs):  # type: ignore
        cached = False
        hashed_key, result = self._cache_lookup("Completion", **kwargs)
        if result is not None:
            cached = True
            if settings.debug:
                print("[grey37]CACHED[/grey37]")
        else:
            # If it's not in the cache, call the API
            limiter = self._rate_limiter()
            raw_call = None
            if self.config.litellm:
                from litellm import completion as litellm_completion

                completion_call = litellm_completion

                if self.api_key != DUMMY_API_KEY:
                    kwargs["api_key"] = self.api_key
            else:
                if self.client is None:
                    raise ValueError("OpenAI/equivalent chat-completion client not set")
                completion_call = self.client.chat.completions.create
                if limiter is not None and isinstance(self.client, OpenAI):
                    # Use the raw-response form so we keep the rate-limit
                    # headers, which the parsed-body form discards.
                    raw_call = self.client.chat.completions.with_raw_response.create
            if self.config.litellm and settings.debug:
                kwargs["logger_fn"] = litellm_logging_fn
            if limiter is not None:
                limiter.acquire()
            try:
                if raw_call is not None:
                    raw_response = raw_call(**kwargs)
                    assert limiter is not None
                    limiter.observe_response(headers=raw_response.headers)
                    result = raw_response.parse()
                else:
                    result = completion_call(**kwargs)
            except Exception as e:
                if limiter is not None:
                    headers = rate_limit_error_headers(e)
                    if headers is not None:
                        limiter.observe_rate_limit_error(headers)
                raise
            if limiter is not None:
                limiter.observe_response(tokens_used=self._response_tokens(result))

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
                    if limiter is not None:
                        headers = rate_limit_error_headers(e)
                        if headers is not None:
                            limiter.observe_rate_limit_error(headers)
                    raise e
            else:
                self._cache_store(hashed_key, result.model_dump())
        return cached, hashed_key, result

    def _chat_completions_with_backoff(self, **kwargs):  # type: ignore
        retry_func = retry_with_exponential_backoff(
            self._chat_completions_with_backoff_body,
            initial_delay=self.config.retry_params.initial_delay,
            max_retries=self.config.retry_params.max_retries,
            exponential_base=self.config.retry_params.exponential_base,
            jitter=self.config.retry_params.jitter,
        )
        return retry_func(**kwargs)

    async def _achat_completions_with_backoff_body(self, **kwargs):  # type: ignore
        cached = False
        hashed_key, result = self._cache_lookup("Completion", **kwargs)
        if result is not None:
            cached = True
            if settings.debug:
                print("[grey37]CACHED[/grey37]")
        else:
            limiter = self._rate_limiter()
            raw_call = None
            if self.config.litellm:
                from litellm import acompletion as litellm_acompletion

                acompletion_call = litellm_acompletion

                if self.api_key != DUMMY_API_KEY:
                    kwargs["api_key"] = self.api_key
            else:
                if self.async_client is None:
                    raise ValueError(
                        "OpenAI/equivalent async chat-completion client not set"
                    )
                acompletion_call = self.async_client.chat.completions.create
                if limiter is not None and isinstance(self.async_client, AsyncOpenAI):
                    # Use the raw-response form so we keep the rate-limit
                    # headers, which the parsed-body form discards.
                    raw_call = (
                        self.async_client.chat.completions.with_raw_response.create
                    )
            if self.config.litellm and settings.debug:
                kwargs["logger_fn"] = litellm_logging_fn
            # If it's not in the cache, call the API
            if limiter is not None:
                await limiter.acquire_async()
            try:
                if raw_call is not None:
                    raw_response = await raw_call(**kwargs)
                    assert limiter is not None
                    limiter.observe_response(headers=raw_response.headers)
                    result = raw_response.parse()
                else:
                    result = await acompletion_call(**kwargs)
            except Exception as e:
                if limiter is not None:
                    headers = rate_limit_error_headers(e)
                    if headers is not None:
                        limiter.observe_rate_limit_error(headers)
                raise
            if limiter is not None:
                limiter.observe_response(tokens_used=self._response_tokens(result))
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
                    if limiter is not None:
                        headers = rate_limit_error_headers(e)
                        if headers is not None:
                            limiter.observe_rate_limit_error(headers)
                    raise e
            else:
                self._cache_store(hashed_key, result.model_dump())
        return cached, hashed_key, result

    async def _achat_completions_with_backoff(self, **kwargs):  # type: ignore
        retry_func = async_retry_with_exponential_backoff(
            self._achat_completions_with_backoff_body,
            initial_delay=self.config.retry_params.initial_delay,
            max_retries=self.config.retry_params.max_retries,
            exponential_base=self.config.retry_params.exponential_base,
            jitter=self.config.retry_params.jitter,
        )
        return await retry_func(**kwargs)

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
                m.api_dict(
                    self.config.chat_model,
                    has_system_role=self.info().allows_system_message,
                )
                for m in (llm_messages)
            ],
            max_completion_tokens=max_tokens,
            stream=self.get_stream(),
        )
        if self.get_stream() and "groq" not in self.chat_model_orig:
            # groq fails when we include stream_options in the request
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
                    functions=[f.model_dump() for f in functions],
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
                            function=t.function.model_dump()
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
        # Only apply allowlist restrictions for known models.
        # Unknown/custom models are allowed to use all params by default.
        is_known_model = self.info().name != "unknown"
        extra_params = args.get("extra_body", {})
        if extra_params and is_known_model:
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
        choices = response.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get("message", {})
        else:
            message = {}
        if message is None:
            message = {}
        content = message.get("content")
        reasoning = message.get("reasoning_content", "")
        # Track whether we extracted inline thought tags from the text.
        # Only set message_with_reasoning when get_reasoning_final()
        # actually finds and extracts inline tags (e.g. <think>...</think>).
        # When reasoning is already provided via a separate API field
        # (e.g. reasoning_content), the message text doesn't contain
        # thought signatures, so there's nothing extra to preserve.
        message_with_reasoning = None
        if reasoning == "" and content is not None:
            # some LLM APIs may not return a separate reasoning field,
            # and the reasoning may be included in the message content
            # within delimiters like <think> ... </think>
            reasoning, msg = self.get_reasoning_final(content)
            if reasoning:
                # Inline tags were found and extracted; preserve the
                # original text so it can be restored in message history.
                message_with_reasoning = content
        else:
            msg = content

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
                    serialized_tool_call = json.dumps(tool_call_dict)
                    msg = (msg or "") + ("\n" if msg else "") + serialized_tool_call
        return LLMResponse(
            # None (no content, e.g. a tool-call-only response) is preserved as
            # None rather than coerced to "", so the distinction survives
            # downstream (see LLMMessage.content / ChatDocument.content_is_none).
            message=msg.strip() if msg is not None else None,
            reasoning=reasoning.strip() if reasoning is not None else "",
            message_with_reasoning=message_with_reasoning,
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
        cached, hashed_key, response = self._chat_completions_with_backoff(**args)  # type: ignore
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
        cached, hashed_key, response = await self._achat_completions_with_backoff(  # type: ignore
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
