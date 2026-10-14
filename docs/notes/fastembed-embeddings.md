# FastEmbed embedding variants

Install the optional dependency with `pip install 'langroid[fastembed]'`
(or `uv sync --dev --extra fastembed` in a development checkout).
The existing FastEmbed dependency range supports all three variants.

| Configuration | Output per text | Similarity |
| --- | --- | --- |
| `FastEmbedEmbeddingsConfig` | Dense vector | Cosine |
| `FastEmbedSparseEmbeddingsConfig` | `SparseEmbedding(indices, values)` | Sparse dot product |
| `FastEmbedLateInteractionEmbeddingsConfig` | Matrix of token vectors | ColBERT MaxSim |

All three are created through `EmbeddingModel.create`. They accept
`model_name`, `batch_size`, `parallel`, `cache_dir`, `threads`, and
`additional_kwargs`. Models are downloaded on first use and can be cached
with `cache_dir`.

```python
from langroid.embedding_models import (
    EmbeddingModel,
    FastEmbedSparseEmbeddingsConfig,
    FastEmbedLateInteractionEmbeddingsConfig,
)

sparse = EmbeddingModel.create(FastEmbedSparseEmbeddingsConfig())
documents = sparse.embedding_fn()(["A document about materials engineering"])
queries = sparse.query_embedding_fn()(["materials"])
print(documents[0].indices, documents[0].values)

late = EmbeddingModel.create(FastEmbedLateInteractionEmbeddingsConfig())
matrices = late.embedding_fn()(["A short passage", "A longer passage of text"])
print(late.embedding_dims)  # Width of each token vector.
print(late.similarity("materials", "A document about materials engineering"))
```

For the concrete sparse model, the callable returns `SparseEmbeddings`;
for the concrete late-interaction model it returns `MultiVectorEmbeddings`.
The existing dense `Embedding`, `Embeddings`, and `EmbeddingFunction` aliases
retain their meanings. Sparse indices and values are converted to Python
lists; token matrices retain both axes and variable token counts.

Use `embedding_fn()` for documents and `query_embedding_fn()` for queries.
Sparse and ColBERT models can encode these differently. `similarity(query,
document)` uses the corresponding encoders. Sparse scores are dot products,
and MaxSim sums each query token's best document-token dot product; neither
score is a bounded cosine similarity. MaxSim is directional.

Sparse models have no fixed dense dimension: accessing `embedding_dims`
raises `ValueError`. For late-interaction models it is the token-vector width,
not the number of tokens. Empty input batches return an empty list without
running either new encoder.

These adapters extend `EmbeddingModel`; they do not add sparse-only or
multivector retrieval to Langroid's vector stores. The current `VectorStore`
interface expects one dense vector per text and rejects these variants
with an explicit error. Use their outputs directly with an appropriately
configured backend. QdrantDB's existing optional SPLADE hybrid path is
unchanged.
