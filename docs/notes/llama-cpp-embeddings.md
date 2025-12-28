# Local embeddings provision via llama.cpp server

As of Langroid v0.30.0, you can use llama.cpp as provider of embeddings
to any of Langroid's vector stores, allowing access to a wide variety of
GGUF-compatible embedding models, e.g.
[nomic-ai's Embed Text V1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF).

## Supported Models

llama.cpp can generate embeddings from:

**Dedicated embedding models (RECOMMENDED):**

- [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
  (768 dims)
- [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF)
- [nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code-GGUF)
- Other GGUF embedding models

**Regular LLMs (also supported):**

- gpt-oss-20b, gpt-oss-120b
- Llama models
- Other language models

Note: Dedicated embedding models are recommended for best performance in
retrieval and semantic search tasks.

## Configuration

When defining a VecDB, you can provide an instance of
`LlamaCppServerEmbeddingsConfig` to the VecDB config to instantiate
the llama.cpp embeddings server handler.

To configure the `LlamaCppServerEmbeddingsConfig`, there are several
parameters that should be adjusted:

```python
from langroid.embedding_models.models import LlamaCppServerEmbeddingsConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

embed_cfg = LlamaCppServerEmbeddingsConfig(
    api_base="http://localhost:8080",  # IP + Port
    dims=768,  # Match the dimensions of your embedding model
    context_length=2048,  # Match the config of the model
    batch_size=2048,  # Safest to ensure this matches context_length
)

vecdb_config = QdrantDBConfig(
    collection_name="my-collection",
    embedding=embed_cfg,
    storage_path=".qdrant/",
)
```

## Running llama-server

The llama.cpp server must be started with the `--embeddings` flag to enable
embedding generation.

### For dedicated embedding models (RECOMMENDED):

```bash
./llama-server -ngl 100 -c 2048 \
  -m ~/nomic-embed-text-v1.5.Q8_0.gguf \
  --host localhost --port 8080 \
  --embeddings -b 2048 -ub 2048
```

### For LLM-based embeddings (e.g., gpt-oss):

```bash
./llama-server -ngl 99 \
  -m ~/.cache/llama.cpp/gpt-oss-20b.gguf \
  --host localhost --port 8080 \
  --embeddings
```

## Response Format Compatibility

Langroid automatically handles multiple llama.cpp response formats:

- Native `/embedding`: `{"embedding": [floats]}`
- OpenAI `/v1/embeddings`: `{"data": [{"embedding": [floats]}]}`
- Array formats: `[{"embedding": [floats]}]`
- Nested formats: `{"embedding": [[floats]]}`

You don't need to worry about which endpoint or format your llama.cpp server
uses - Langroid will automatically detect and handle the response correctly.

## Example Usage

An example setup can be found inside
[examples/docqa/chat.py](https://github.com/langroid/langroid/blob/main/examples/docqa/chat.py).

For a complete example using local embeddings with llama.cpp:

```python
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.embedding_models.models import LlamaCppServerEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

# Configure local embeddings via llama.cpp
embed_cfg = LlamaCppServerEmbeddingsConfig(
    api_base="http://localhost:8080",
    dims=768,  # nomic-embed-text-v1.5 dimensions
    context_length=8192,
    batch_size=1024,
)

# Configure vector store with local embeddings
vecdb_config = QdrantDBConfig(
    collection_name="doc-chat-local",
    embedding=embed_cfg,
    storage_path=".qdrant/",
)

# Create DocChatAgent
config = DocChatAgentConfig(
    vecdb=vecdb_config,
    llm=OpenAIGPTConfig(
        chat_model="gpt-4o",  # or use local LLM
    ),
)

agent = DocChatAgent(config)
```

## Troubleshooting

**Error: "Failed to connect to embedding provider"**

- Ensure llama-server is running with the `--embeddings` flag
- Check that the `api_base` URL is correct
- Verify the server is accessible from your machine

**Error: "Unsupported embedding response format"**

- This error includes the first 500 characters of the response to help debug
- Check your llama-server logs for any errors
- Ensure you're using a compatible llama.cpp version

**Embeddings seem low quality:**

- Use a dedicated embedding model instead of an LLM
- Ensure the `dims` parameter matches your model's output dimensions
- Try different GGUF quantization levels (Q8_0 generally works well)

## Additional Resources

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)
- [nomic-embed models on Hugging Face](https://huggingface.co/nomic-ai)
- [Issue #919 - Implementation details](https://github.com/langroid/langroid/blob/main/issues/issue-919-llamacpp-embeddings.md)

