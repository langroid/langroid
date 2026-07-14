# Using DeepLake as a Vector Store with Langroid

[DeepLake](https://github.com/activeloopai/deeplake) is a vector store that runs
locally out of the box, so unlike cloud stores there is no account or API key
needed to get started.

## Installation

Install Langroid with the `deeplake` extra:

```
uv add langroid[deeplake]     # or: pip install langroid[deeplake]
```

## Storage

By default each collection is stored locally under `.deeplake/data`, one
sub-directory per collection. Set `storage_path` to point elsewhere, including
an Activeloop-hosted `hub://<org>/<name>` path. For hosted datasets, set your
token in the `.env` file:

```env
ACTIVELOOP_TOKEN=<your_token>
```

## Code Example

```python
import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import OpenAIEmbeddingsConfig

embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
    vecdb=lr.vector_store.DeepLakeDBConfig(
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

agent = DocChatAgent(config)
```

## Ingest Documents and Query

```python
documents = [
    lr.Document(
        content="""
            In the year 2050, GPT10 was released.
            In 2057, paperclips were seen all over the world.
            Global warming was solved in 2060.
            In 2061, the world was taken over by paperclips.
        """,
        metadata=lr.DocMetaData(source="wikipedia-2063", id="doc-1"),
    ),
]

agent.ingest_docs(documents)
answer = agent.llm_response("When was global warming solved?")
```
