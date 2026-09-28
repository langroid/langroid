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
Langroid can construct this URL and refresh Google Application Default
Credentials (ADC) automatically when the model uses the
`vertexai/<publisher>/<model>` format.

### First-class Vertex AI route

Authenticate ADC and identify the Google Cloud project and location:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

Then prefix the publisher-qualified model name with `vertexai/`:

```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(
    chat_model="vertexai/google/gemini-3-flash",
)
llm = lm.OpenAIGPT(config)
response = llm.chat("Hello from Vertex AI!")
```

The project and location can instead be specified directly in the config:

```python
config = lm.OpenAIGPTConfig(
    chat_model="vertexai/google/gemini-3-flash",
    vertexai_project_id="my-gcp-project",
    vertexai_location="us-central1",
)
```

For the `global` location, Langroid uses `aiplatform.googleapis.com`; regional
locations use `<location>-aiplatform.googleapis.com`. An explicit `api_base`
takes precedence over the generated endpoint. A caller-supplied
`api_key_provider` or `api_key` also takes precedence over automatic ADC.

### Manual endpoint configuration

The existing manual configuration remains available. Generate a short-lived
access token and provide the full endpoint URL:

```bash
export GEMINI_API_KEY=$(gcloud auth print-access-token)
export GEMINI_API_BASE=https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-gcp-project/locations/us-central1/endpoints/openapi
```

```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(chat_model="gemini/gemini-2.0-flash")
llm = lm.OpenAIGPT(config)
response = llm.chat("Hello from Vertex AI!")
```

!!! note
    The `OPENAI_API_BASE` environment variable (commonly used for local
    proxies) is **not** applied to Gemini models. For the manual `gemini/`
    route, use `GEMINI_API_BASE` or an explicit `api_base` instead.

When neither `GEMINI_API_BASE` nor an explicit `api_base` is set, Langroid
falls back to the default Google AI (Gemini) endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`).
