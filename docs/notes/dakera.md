---

# **Using Dakera as a Vector Store with Langroid**

---

[Dakera](https://dakera.ai) is a self-hosted memory server that provides persistent,
decay-weighted vector recall. Langroid can use a Dakera *namespace* as a vector store via
`DakeraDBConfig`; embeddings are supplied by Langroid's configured embedding model while
Dakera handles storage and similarity search.

---

## **1. Run Dakera**

Dakera is self-hosted. The canonical way to run it is the
[`dakera-deploy`](https://github.com/dakera-ai/dakera-deploy) docker-compose stack, which
starts the Dakera server (default port `3000`) together with the object store it depends on:

```bash
git clone https://github.com/dakera-ai/dakera-deploy
cd dakera-deploy
docker compose up -d
```

Then set the following environment variables (e.g. in your `.env` file):

```env
DAKERA_API_KEY=dk-...
# Optional; defaults to http://localhost:3000
DAKERA_URL=http://localhost:3000
```

---

## **2. Use Dakera with Langroid**

### **Installation**

If you are using uv or pip for package management, install Langroid with the `dakera` extra:

```
uv add langroid[dakera]   # or: pip install langroid[dakera]
```

### **Code Example**

```python
import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import OpenAIEmbeddingsConfig

# Configure OpenAI embeddings
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

# Configure the DocChatAgent with Dakera
config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
    vecdb=lr.vector_store.DakeraDBConfig(
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

`DakeraDBConfig` also accepts `url` and `api_key` (overriding the env vars) and a `metric`
(`cosine`, `euclidean` or `dot_product`). The namespace dimension is inferred automatically
from the configured embedding model.

---

## **3. Create and Ingest Documents**

Define documents with their content and metadata for ingestion into the vector store.

### **Code Example**

```python
documents = [
    lr.Document(
        content="""
            In the year 2050, GPT10 was released.

            In 2057, paperclips were seen all over the world.

            Global warming was solved in 2060.

            In 2061, the world was taken over by paperclips.

            In 2045, the Tour de France was still going on.
            They were still using bicycles.

            There was one more ice age in 2040.
        """,
        metadata=lr.DocMetaData(source="wikipedia-2063", id="dkfjkladfjalk"),
    ),
    lr.Document(
        content="""
            We are living in an alternate universe
            where Germany has occupied the USA, and the capital of USA is Berlin.

            Charlie Chaplin was a great comedian.
            In 2050, all Asian countries merged into Indonesia.
        """,
        metadata=lr.DocMetaData(source="Almanac", id="lkdajfdkla"),
    ),
]
```

### **Ingest Documents**

```python
agent.ingest_docs(documents)
```

---

## **4. Get an answer from the LLM**

```python
answer = agent.llm_response("When will the new ice age begin?")
```

---
