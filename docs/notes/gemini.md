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



