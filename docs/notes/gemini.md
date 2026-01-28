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
To use Gemini models through Vertex AI, specify a custom `api_base` in
`OpenAIGPTConfig`.

### Setup

1. Set up authentication. Vertex AI typically uses Google Cloud credentials
   rather than a simple API key. You can generate a short-lived access token:

    ```bash
    export GEMINI_API_KEY=$(gcloud auth print-access-token)
    ```

2. Identify your Vertex AI endpoint URL, which includes your GCP project ID
   and region:

    ```
    https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi
    ```

### Usage

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

When `api_base` is not set, Langroid falls back to the default Google AI
(Gemini) endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`).
