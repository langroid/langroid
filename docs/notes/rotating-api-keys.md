# Rotating / Short-Lived API Keys

As of version 0.65.14, `OpenAIGPTConfig` supports short-lived, rotating
credentials via the `api_key_provider` field: a callable that returns a
currently-valid API key or bearer token.

## The Problem

Some OpenAI-compatible endpoints do not authenticate with a permanent API key.
Instead they expect a short-lived bearer token that must be periodically
refreshed:

- **Google Vertex AI** uses OAuth / Application Default Credentials (ADC)
  tokens, which typically expire after about an hour.
- **Azure OpenAI with Microsoft Entra ID** uses Entra ID access tokens.

Neither of the obvious workarounds is satisfactory:

- Passing the token as a static `api_key` bakes it into the underlying client
  at construction time. Once the token expires, every subsequent request fails
  with a 401, and the only fix is to rebuild the LLM object.
- Rebuilding the config with a fresh token on each call works, but each new
  token value produces a new client-cache entry, so the process-wide client
  cache grows without bound.

`api_key_provider` solves both: the callable is resolved before every request,
and the client cache is keyed on the callable itself rather than on any token
value.

## Basic Usage

Supply a plain synchronous function that returns a valid key each time it is
called:

```python
import langroid.language_models as lm

def get_token() -> str:
    ...  # fetch/refresh and return a currently-valid token

config = lm.OpenAIGPTConfig(
    chat_model="...",
    api_base="...",
    api_key_provider=get_token,
)

llm = lm.OpenAIGPT(config)
```

Your provider is responsible for caching and refreshing: it is called on every
request, so it should return a cached token when the current one is still
valid, and refresh only when needed.

## Example: Google Vertex AI

The canonical use case. Point `api_base` at Vertex AI's OpenAI-compatible
endpoint, and use `google-auth` to mint and refresh the ADC token:

```python
import google.auth
import google.auth.transport.requests
import langroid.language_models as lm

creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

def vertex_token() -> str:
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    return creds.token

config = lm.OpenAIGPTConfig(
    chat_model="google/gemini-2.5-flash",
    api_base=(
        "https://aiplatform.googleapis.com/v1beta1/projects/<project>"
        "/locations/global/endpoints/openapi"
    ),
    api_key_provider=vertex_token,
)

llm = lm.OpenAIGPT(config)
```

The `creds` object caches the token internally, so `vertex_token()` only makes
a network call when the token has actually expired.

## How It Works

- **Resolved per request.** The callable is handed directly to the OpenAI
  Python SDK client (`OpenAI(api_key=<callable>)`), which resolves it before
  every request, including streaming requests and retries. Tokens therefore
  never go stale, no matter how long the process runs. This relies on the
  SDK's native support for callable API keys, introduced in `openai` 1.106.0
  (Langroid's dependency floor guarantees this).
- **Async is handled for you.** `AsyncOpenAI` awaits its `api_key` callable,
  so Langroid automatically wraps your synchronous provider in an async
  wrapper. You always supply a plain sync callable returning `str`, and both
  `llm.chat(...)` and `llm.achat(...)` work.
- **The provider is excluded from the client-cache key.** The process-wide
  client cache is keyed on the provider's *identity* (`id()`), never on a
  token value. Rotating tokens thus create no new cache entries, and token
  values are never retained in cache keys. Two `OpenAIGPT` instances that
  share the same provider callable share one cached client.
- **Precedence.** When `api_key_provider` is set, it takes precedence over
  `api_key` and over the `OPENAI_API_KEY` environment variable.

It works with `use_cached_client=True` (the default) and with
`use_cached_client=False`, and composes with `http_client_factory` and
`http_client_config` — see
[OpenAI HTTP Client Configuration](openai-http-client.md) and
[OpenAI Client Caching](openai-client-caching.md).

## Limitations

`api_key_provider` is only supported for models served via an
OpenAI-compatible endpoint, i.e. those that go through the OpenAI client path.
This includes plain OpenAI models as well as the `gemini/`, `litellm-proxy/`,
`openrouter/`, `deepseek/`, `vllm/`, and `local/` prefixes.

Constructing an `OpenAIGPT` with `api_key_provider` raises a `ValueError` for:

- `groq/...` models, which use the Groq SDK client
- `cerebras/...` models, which use the Cerebras SDK client
- the litellm adapter library, i.e. `litellm=True` or `litellm/...` models

## Alternative: Transport-Level Auth with `http_client_factory`

If you need more control than a bearer token string — custom headers, request
signing (e.g. AWS SigV4 for Bedrock-style auth), or an asynchronous token
fetch — you can instead attach an `httpx.Auth` to a custom HTTP client. The
auth flow then runs at the transport layer, on every request:

```python
import httpx
import google.auth
import google.auth.transport.requests
import langroid.language_models as lm

class _BearerRefresh(httpx.Auth):
    def __init__(self):
        self._creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def auth_flow(self, request):
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self._creds.token}"
        yield request

config = lm.OpenAIGPTConfig(
    chat_model="google/gemini-2.5-flash",
    api_base="...",
    http_client_factory=lambda: (
        httpx.Client(auth=_BearerRefresh()),
        httpx.AsyncClient(auth=_BearerRefresh()),
    ),
)
```

The factory may return either a single `httpx.Client` or a tuple of
`(httpx.Client, httpx.AsyncClient)`. Returning a single client attaches your
auth flow to synchronous calls (`llm.chat(...)`) only; the tuple form, as
above, covers asynchronous calls (`llm.achat(...)`) as well.

Note that a factory-supplied client bypasses the client cache: the
`OpenAIGPT` instance builds its own client around it. But if the factory
returns only a sync client, the async client is still created through the
normal cached path, using the static `api_key` — so `llm.achat(...)` would
not carry your transport-level auth. Prefer `api_key_provider` unless you
specifically need transport-level control.

## Azure OpenAI with Entra ID

For Azure-deployed models (`AzureConfig` / `AzureGPT`), use the existing
`azure_openai_client_provider` and `azure_openai_async_client_provider` config
fields to supply clients built with the SDK's native `azure_ad_token_provider`:

```python
import langroid.language_models as lm
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

lm_config = lm.AzureConfig(
    azure_openai_client_provider=lambda: AzureOpenAI(
        azure_ad_token_provider=token_provider,
        # ... azure_endpoint, api_version, etc.
    ),
    azure_openai_async_client_provider=lambda: AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        # ... azure_endpoint, api_version, etc.
    ),
)
```

See [Custom Azure OpenAI client](custom-azure-client.md) for details.

## See Also

- [OpenAI HTTP Client Configuration](openai-http-client.md) — proxies,
  custom CA bundles, and custom HTTP clients
- [OpenAI Client Caching](openai-client-caching.md) — how clients are cached
  and shared
