# LangDB with Langroid

## Introduction

[LangDB](https://langdb.ai/) is an AI gateway that provides OpenAI-compatible APIs to access 250+ LLMs. It offers cost control, observability, and performance benchmarking while enabling seamless switching between models. 
Langroid has a simple integration with LangDB's API service, so there are no dependencies
to install. (LangDB also has a self-hosted version, which is not yet supported in Langroid).

## Setup environment variables

At minimum, ensure you have these env vars in your `.env` file:

```
LANGDB_API_KEY=your_api_key_here
LANGDB_PROJECT_ID=your_project_id_here
```

## Using LangDB with Langroid

### Configure LLM and Embeddings

In `OpenAIGPTConfig`, when you specify the `chat_model` with a `langdb/` prefix,
langroid uses the API key, `project_id` and other langDB-specific parameters
from the `langdb_params` field; if any of these are specified in the `.env` file
or in the environment explicitly, they will override the values in `langdb_params`.
For example, to use Anthropic's Claude-3.7-Sonnet model, 
set `chat_model="langdb/anthropic/claude-3.7-sonnet":

```python
import os
import uuid
from langroid.language_models.openai_gpt import OpenAIGPTConfig, LangDBParams
from langroid.embedding_models.models import OpenAIEmbeddingsConfig

# Generate tracking IDs (optional)
thread_id = str(uuid.uuid4())
run_id = str(uuid.uuid4())

# Configure LLM
llm_config = OpenAIGPTConfig(
    chat_model="langdb/openai/gpt-4",  # LangDB model prefix
    langdb_params=LangDBParams(
        label='my-app',
        thread_id=thread_id,
        run_id=run_id,
        # api_key, project_id are read from .env or environment variables
        # LANGDB_API_KEY, LANGDB_PROJECT_ID respectively.
    )
)
```

Similarly, you can configure the embeddings using `OpenAIEmbeddingsConfig`,
which also has a `langdb_params` field that works the same way as 
in `OpenAIGPTConfig` (i.e. it uses the API key and project ID from the environment
if provided, otherwise uses the default values in `langdb_params`):

```python
# Configure embeddings
embedding_config = OpenAIEmbeddingsConfig(
    model_name="langdb/openai/text-embedding-3-small",
)
```

## Tracking and Observability

LangDB provides special headers for request tracking:

- `x-label`: Tag requests for filtering in the dashboard
- `x-thread-id`: Track conversation threads (UUID format)
- `x-run-id`: Group related requests together

## Examples

The `langroid/examples/langdb/` directory contains examples demonstrating:

1. **RAG with LangDB**: `langdb_chat_agent_docs.py`
2. **LangDB with Function Calling**: `langdb_chat_agent_tool.py`
3. **Custom Headers**: `langdb_custom_headers.py`

## Viewing Results

Visit the [LangDB Dashboard](https://dashboard.langdb.com) to:
- Filter requests by label, thread ID, or run ID
- View detailed request/response information
- Analyze token usage and costs

For more information, visit [LangDB Documentation](https://docs.langdb.com).