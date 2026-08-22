# Using Milvus as a Vector Store with Langroid

Langroid supports Milvus through `MilvusDBConfig` and the PyMilvus `MilvusClient`.
The default configuration uses Milvus Lite at `./milvus.db`, so no separate
service is required for a local start.

## Installation

Install Langroid with the Milvus extra:

```bash
uv add "langroid[milvus]"
```

You can also install it with `pip`:

```bash
pip install "langroid[milvus]"
```

## Platform support

Milvus Lite is not available on Windows. Windows users must set `MILVUS_URI` to
a Milvus server or Zilliz Cloud endpoint.

## Resource lifetime with Milvus Lite

`MilvusDB.close()` releases the PyMilvus client handle, but Milvus Lite keeps
its embedded per-path server (and the lock on the `.db` file) alive until the
process exits — this is upstream PyMilvus behavior. If another process needs
the same `.db` file, close the first process rather than relying on `close()`.

## Configuration

By default, `MilvusDBConfig()` connects to Milvus Lite:

```python
import langroid as lr

vecdb = lr.vector_store.MilvusDBConfig(
    collection_name="quick_start_chat_agent_docs",
    replace_collection=True,
)
```

To use a local or remote Milvus service, pass `uri` explicitly or set
`MILVUS_URI`:

```bash
export MILVUS_URI="http://localhost:19530"
```

For Zilliz Cloud, set the cloud endpoint and token:

```bash
export MILVUS_URI="https://your-project.api.region.zillizcloud.com"
export MILVUS_TOKEN="your-token"
```

If you use a named Milvus database, set `MILVUS_DB_NAME` or pass `db_name` in
the config.

## Example

```python
import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig

config = DocChatAgentConfig(
    vecdb=lr.vector_store.MilvusDBConfig(
        collection_name="quick_start_chat_agent_docs",
        uri="./milvus.db",
        replace_collection=True,
    ),
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.SIMPLE,
    ),
    n_similar_chunks=2,
    n_relevant_chunks=2,
)

agent = DocChatAgent(config)

documents = [
    lr.Document(
        content="Milvus Lite stores vectors in a local file.",
        metadata=lr.DocMetaData(source="milvus-docs", id="milvus-lite"),
    ),
    lr.Document(
        content="The same config can point to Milvus server or Zilliz Cloud.",
        metadata=lr.DocMetaData(source="milvus-docs", id="milvus-service"),
    ),
]

agent.ingest_docs(documents)
```

Milvus stores Langroid document metadata in a JSON field and also exposes scalar
metadata fields for simple filters, for example:

```python
matches = agent.vecdb.similar_texts_with_scores(
    "local vector storage",
    k=2,
    where='{"source": "milvus-docs"}',
)
```
