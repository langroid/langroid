# LangDB Examples

This folder contains examples demonstrating how to use [LangDB](https://langdb.com) with Langroid for advanced LLM observability and monitoring.

## Prerequisites

Before running any examples, make sure you've installed Langroid as usual.


At minimum, have these environment variables set up in your `.env` file or environment:
```bash
LANGDB_API_KEY=your_api_key_here
LANGDB_PROJECT_ID=your_project_id_here
```

### 1. LangDB Chat Agent with Document RAG (`langdb_chat_agent_docs.py`)

Demonstrates Retrieval Augmented Generation (RAG) with LangDB integration:
- Ingests documents into a vector database
- Uses LangDB for both chat completions and embeddings
- Tracks all interactions with custom headers for observability

```python
# Run the example
python langdb_chat_agent_docs.py
```

### 2. LangDB Chat Agent with Tool (`langdb_chat_agent_tool.py`)

Shows how to use LangDB with function-calling capabilities:
- Implements a number-guessing game using tools
- Demonstrates custom header usage for request tracking
- Shows how to integrate LangDB with stateful agents

```python
# Run the example
python langdb_chat_agent_tool.py
```

### 3. LangDB Custom Headers (`langdb_custom_headers.py`)

Showcases LangDB's observability features:
- `x-label`: Tag requests for filtering in the LangDB dashboard
- `x-thread-id`: Track conversation threads (UUID format)
- `x-run-id`: Group related requests together

```python
# Run the example
python langdb_custom_headers.py
```

## Using LangDB

### Configuring LLM and Embeddings

LangDB can be used for both chat completions and embeddings:

```python
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig, LangDBParams
from langroid.vector_store.qdrant import QdrantDBConfig
import os
import uuid

# Generate IDs for request tracking
run_id = str(uuid.uuid4())
thread_id = str(uuid.uuid4())

# Configure LLM with LangDBParams
llm_config = OpenAIGPTConfig(
    chat_model="langdb/openai/gpt-4",  # LangDB model prefix
    langdb_params=LangDBParams(
        label='my-app',
        thread_id=thread_id,  # For conversation tracking
        run_id=run_id,        # For request grouping
        # project_id, api_key are used from the env vars
        # LANGDB_API_KEY, LANGDB_PROJECT_ID respectively
    )
)

# Configure embeddings
vecdb_config = QdrantDBConfig(
    collection_name="my-docs",
    embedding=OpenAIEmbeddingsConfig(
        model_name="langdb/openai/text-embedding-3-small",
        # langdb_params will contain api_key from env var LANGDB_API_KEY
    )
)
```

### Custom Headers

LangDB provides special headers for request tracking through the LangDBParams class:

```python
# Generate a thread ID
import uuid
import os
from langroid.language_models.openai_gpt import OpenAIGPTConfig, LangDBParams

# Generate tracking IDs using UUID
thread_id = str(uuid.uuid4())
run_id = str(uuid.uuid4())  # Use UUID for run_id as well

# Configure with LangDBParams
config = OpenAIGPTConfig(
    chat_model="langdb/openai/gpt-4o-mini",
    langdb_params=LangDBParams(
        label="my-label",
        thread_id=thread_id,
        run_id=run_id,
        # project_id is set via env var LANGDB_PROJECT_ID
        # api_key is set via env var LANGDB_API_KEY
    )
)
```

### Viewing Results

1. Visit the [LangDB Dashboard](https://dashboard.langdb.com)
2. Navigate to your project
3. Use filters to find your requests:
   - Search by label, thread ID, or run ID
   - View detailed request/response information
   - Analyze token usage and costs

## Best Practices

1. **Unique Thread IDs**: Always generate new UUIDs for conversation threads
2. **Descriptive Labels**: Use meaningful labels to identify different parts of your application
3. **Consistent Run IDs**: Group related requests under the same run ID
4. **Environment Variables**: Never hardcode API keys or project IDs

## Troubleshooting

Common issues and solutions:

1. **Authentication Errors**:
   - Verify `LANGDB_API_KEY` is set correctly
   - Check if the key has the necessary permissions

2. **Model Not Found**:
   - Ensure the model name includes the `langdb/` prefix
   - Verify the model is available in your subscription

3. **Header Issues**:
   - Thread IDs must be valid UUIDs
   - Labels should be URL-safe strings

For more help, visit the [LangDB Documentation](https://docs.langdb.com).


```python
# Generate a proper UUID for thread-id
import uuid
import os
from langroid.language_models.openai_gpt import OpenAIGPTConfig, LangDBParams

thread_id = str(uuid.uuid4())
run_id = str(uuid.uuid4())

# Create a LangDB model configuration with LangDBParams
langdb_config = OpenAIGPTConfig(
    chat_model="langdb/openai/gpt-4o-mini",
    langdb_params=LangDBParams(
        label='langroid',
        run_id=run_id,
        thread_id=thread_id,
        # project_id is set via env var LANGDB_PROJECT_ID
        # api_key is set via env var LANGDB_API_KEY
    )
)

# The headers will be automatically added to requests
```

These parameters allow you to track and organize your LangDB requests. While these parameters can be used with any model provider, they are only meaningful when used with LangDB.

**Note**: The `thread_id` and `run_id` parameters must be a valid UUID format. 
The examples use `uuid.uuid4()` to generate a proper UUID.
