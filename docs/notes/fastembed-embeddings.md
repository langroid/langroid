# FastEmbed sparse embeddings for Qdrant

QdrantDB supports FastEmbed's `SparseTextEmbedding` as an alternative to its
existing transformers-based SPLADE encoder. Enable sparse embeddings and select
the `fastembed` backend to use it for both document indexing and sparse queries.
Both backends produce the Qdrant `SparseVector` objects used by the store's
existing sparse search path.

## Installation

```bash
pip install "langroid[fastembed]"
```

For a development checkout:

```bash
uv sync --dev --extra fastembed
```

FastEmbed uses ONNX inference for the sparse model. This sparse encoding path
does not require torch or transformers. The dense embedding model is configured
separately and retains its own dependency and authentication requirements.

Models download on first use, and subsequent initializations can reuse the
FastEmbed model cache.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `use_sparse_embeddings` | `False` | Enable the existing Qdrant sparse search path. |
| `sparse_embedding_backend` | `"transformers"` | Select `"transformers"` or `"fastembed"`. |
| `sparse_embedding_model` | `"naver/splade-v3-distilbert"` | Model for the transformers backend. |
| `fastembed_sparse_model` | `"prithivida/Splade_PP_en_v1"` | Model for the FastEmbed backend; this is the integration-tested SPLADE model. |
| `sparse_limit` | `3` | Maximum number of results requested from the sparse search. |

Existing configurations continue to select transformers when sparse embeddings
are enabled. Selecting FastEmbed uses `fastembed_sparse_model`; it leaves the
transformers model setting unchanged. The FastEmbed encoder uses the store's
existing `batch_size` setting.

Use the same sparse backend and model for indexing and querying a collection.
When changing the sparse model for existing data, re-index the documents into a
fresh collection so stored vectors and query vectors use the same representation.

## Example: local dense and sparse retrieval

This example uses FastEmbed for both dense and sparse embeddings. It needs no
embedding API key or Qdrant server. The two models download on first use.

```python
from langroid.embedding_models.models import FastEmbedEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

config = QdrantDBConfig(
    cloud=False,
    storage_path=":memory:",
    collection_name="fastembed-sparse-example",
    embedding=FastEmbedEmbeddingsConfig(
        model_name="BAAI/bge-small-en-v1.5",
    ),
    use_sparse_embeddings=True,
    sparse_embedding_backend="fastembed",
    fastembed_sparse_model="prithivida/Splade_PP_en_v1",
)

with QdrantDB(config) as db:
    db.add_documents(
        [
            Document(
                content="The rocket launches a satellite into orbit.",
                metadata=DocMetaData(),
            ),
            Document(
                content="Fresh bananas are sold at the fruit market.",
                metadata=DocMetaData(),
            ),
        ]
    )
    for document, score in db.similar_texts_with_scores("rocket", k=1):
        print(score, document.content)
```

`storage_path=":memory:"` creates an in-memory Qdrant instance whose data is
discarded when the client closes. Set `storage_path` to a directory for persistent
local storage, or use the existing Qdrant cloud configuration.

## Encoding and search behavior

Document ingestion calls FastEmbed's `embed()` method. Queries call
`query_embed()`, allowing the model to apply its query-specific encoding.
Sparse indices and weights are converted to Python lists and stored in the
existing `"text-sparse"` vector field alongside dense vectors.

Retrieval keeps QdrantDB's existing combination of dense and sparse search
results. Sparse scores are dot products and have a different scale from dense
cosine scores. This backend does not introduce a new score-fusion strategy.

The public `EmbeddingModel`, `embedding_fn()`, and dense embedding type aliases
retain their existing contracts. Late-interaction and multivector retrieval are
outside this change.

## Tests

Offline tests use deterministic encoder outputs and real in-memory Qdrant to
check conversion, query routing, storage, retrieval, empty vectors, and optional
dependency errors. They do not download models:

```bash
uv run pytest -q tests/main/test_qdrant_fastembed_sparse.py
```

The real FastEmbed SPLADE test is kept under `tests/extras` because it downloads
a model on first execution:

```bash
uv run --extra fastembed pytest -q tests/extras/test_qdrant_fastembed_sparse_integration.py
```

To run both files against the declared minimum FastEmbed version:

```bash
uv run --extra fastembed --with "fastembed==0.3.1" pytest -q tests/main/test_qdrant_fastembed_sparse.py tests/extras/test_qdrant_fastembed_sparse_integration.py
```
