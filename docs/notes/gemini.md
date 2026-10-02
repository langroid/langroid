# Gemini LLMs & Embeddings via OpenAI client (without LiteLLM)

As of Langroid v0.21.0 you can use Langroid with Gemini LLMs directly
via the OpenAI client, without using adapter libraries like LiteLLM.

See details [here](https://langroid.github.io/langroid/tutorials/non-openai-llms/)

You can use also Google AI Studio Embeddings or Gemini Embeddings directly
which uses google-generativeai client under the hood.

```python

import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import GeminiEmbeddingsConfig

# Configure Gemini embeddings
embed_cfg = GeminiEmbeddingsConfig(
    model_type="gemini",
    model_name="models/text-embedding-004",
    dims=768,
)

# Configure the DocChatAgent
config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model="gemini/" + lr.language_models.GeminiModel.GEMINI_1_5_FLASH_8B,
    ),
    vecdb=lr.vector_store.QdrantDBConfig(
        collection_name="quick_start_chat_agent_docs",
        replace_collection=True,
        embedding=embed_cfg,
    ),
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.SIMPLE,
    ),
    n_similar_chunks=2,
    n_relevant_chunks=2,
)

# Create the agent
agent = DocChatAgent(config)
```

## Vertex AI Support

Google Vertex AI uses project-specific URLs for its
[OpenAI compatibility layer](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library),
which differs from the fixed URL used by the standard Google AI (Gemini) API.
To use Gemini models through Vertex AI, set the endpoint via the
`GEMINI_API_BASE` environment variable or the `api_base` parameter in
`OpenAIGPTConfig`.

!!! note
    The `OPENAI_API_BASE` environment variable (commonly used for local
    proxies) is **not** applied to Gemini models. Use `GEMINI_API_BASE`
    or an explicit `api_base` in the config instead.

### Setup

1. Set up authentication. Vertex AI typically uses Google Cloud credentials
   rather than a simple API key. You can generate a short-lived access token:

    ```bash
    export GEMINI_API_KEY=$(gcloud auth print-access-token)
    ```

2. Set your Vertex AI endpoint URL, which includes your GCP project ID
   and region:

    ```bash
    export GEMINI_API_BASE=https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi
    ```

### Usage

**Option 1: Environment variable (recommended for Vertex AI)**

```bash
export GEMINI_API_KEY=$(gcloud auth print-access-token)
export GEMINI_API_BASE=https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-gcp-project/locations/us-central1/endpoints/openapi
```

```python
import langroid.language_models as lm

# GEMINI_API_BASE is picked up automatically
config = lm.OpenAIGPTConfig(chat_model="gemini/gemini-2.0-flash")
llm = lm.OpenAIGPT(config)
response = llm.chat("Hello from Vertex AI!")
```

**Option 2: Explicit `api_base` in config**

```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(
    chat_model="gemini/gemini-2.0-flash",
    api_base=(
        "https://us-central1-aiplatform.googleapis.com/v1beta1"
        "/projects/my-gcp-project/locations/us-central1/endpoints/openapi"
    ),
)
llm = lm.OpenAIGPT(config)
response = llm.chat("Hello from Vertex AI!")
```

When neither `GEMINI_API_BASE` nor an explicit `api_base` is set, Langroid
falls back to the default Google AI (Gemini) endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`).

### Keeping the Vertex AI token fresh

`gcloud auth print-access-token` returns a token that expires in about an
hour, so a long-running agent configured from the environment will start
failing mid-run. Rather than restarting it, hand Langroid a callable and it
will fetch a token for every request:

```python
import subprocess

import langroid.language_models as lm


def vertex_token() -> str:
    return subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


config = lm.OpenAIGPTConfig(
    chat_model="gemini/gemini-2.0-flash",
    api_base=(
        "https://us-central1-aiplatform.googleapis.com/v1beta1"
        "/projects/my-gcp-project/locations/us-central1/endpoints/openapi"
    ),
    api_key_provider=vertex_token,
)
```

`api_key_provider` takes precedence over `api_key`, and the provider is
called per request, so rotation is automatic. See
[Rotating / Short-Lived API Keys](rotating-api-keys.md) for its full
behaviour, including the async form and client caching.

The official credential libraries cache the token instead, and refresh it
only once it has expired:

```python
import google.auth
import google.auth.transport.requests

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
request = google.auth.transport.requests.Request()


def vertex_token() -> str:
    if not credentials.valid:
        credentials.refresh(request)
    return credentials.token
```

That needs `google-auth`, which Langroid does not depend on; install it
yourself if you use this form. Prefer it over the `gcloud` form for anything
long-running: shelling out spawns a process on *every* request, whereas the
credential libraries cache the token and refresh it only once it has actually
expired.

### Headers set for OpenAI follow you to Vertex AI

`OpenAIGPTConfig` is a pydantic `BaseSettings` whose env prefix is
`OPENAI_`, so matching environment variables populate the corresponding
fields whichever model you then point the config at. For `api_key` there is
a guard: when the model is recognized as Gemini (a `gemini/` or
`google/gemini-` prefix) and
`api_key` still holds what `OPENAI_API_KEY` put there, Langroid replaces it
with `GEMINI_API_KEY`, or with a dummy key if that is unset — so the OpenAI
key is not sent to Google, and a config relying on it fails to authenticate
instead.

**`headers` has no such guard.** With `OPENAI_HEADERS` set, those headers
are sent to
`googleapis.com` along with your request, and if the dict contains an
`Authorization` key it *replaces* the Gemini token, so the call both leaks
a credential to Google and fails to authenticate. `OPENAI_ORGANIZATION`
reaches Google the same way. `api_base` is the other exception: the Gemini
route ignores `OPENAI_API_BASE`, as noted above.

Passing `headers={}` to the constructor does **not** prevent this:
pydantic-settings merges the environment dict into the one you pass, so
your keys win individually but the rest still ride along. Clear the field
after the config is built:

```python
config = lm.OpenAIGPTConfig(
    chat_model="gemini/gemini-2.0-flash",
    api_base="https://us-central1-aiplatform.googleapis.com/v1beta1/...",
    api_key_provider=vertex_token,
)
config.headers = {}        # drop anything inherited from OPENAI_HEADERS
llm = lm.OpenAIGPT(config)
```

Two further cases where the OpenAI *key* does reach Google, both worth
knowing if you set that environment up yourself:

- `chat_model="gemini-2.0-flash"` without the `gemini/` prefix is not
  treated as a Gemini model at all, so `api_key` keeps the OpenAI value
  while `api_base` still points at Google. Always keep the prefix.
- A lower-cased `openai_api_key` in the environment populates the field
  (pydantic matches case-insensitively) but is missed by the Gemini
  route's own lookup. Use the upper-case spelling.
