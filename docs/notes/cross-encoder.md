# Cross-Encoder Reranking

`DocChatAgent` can retrieve candidate chunks with several methods (semantic search, BM25, fuzzy
matching) and then re-order them. A cross-encoder produces the most accurate ordering of the three,
because it scores each `(query, passage)` pair jointly instead of comparing embeddings, but it is
also the slowest: scoring is a model call per passage.

Cross-encoder re-ranking is optional and needs `sentence-transformers` and `torch`, which the
`hf-embeddings` extra installs:

```bash
pip install "langroid[hf-embeddings]"
# or
uv add "langroid[hf-embeddings]"
```

## Configuration

`DocChatAgentConfig` chooses between the cross-encoder and
[Reciprocal Rank Fusion](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking#how-rrf-ranking-works)
(RRF), which fuses the ranked lists produced by the individual retrieval methods:

| Setting | Default | Effect |
| --- | --- | --- |
| `cross_encoder_reranking_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` when `sentence-transformers` is importable, otherwise `""` | Model used to score `(query, passage)` pairs. The empty default means "no cross-encoder re-ranking". |
| `cross_encoder_device` | `None` | `None` selects `cuda`, then `mps`, then `cpu`, in that order. Set it to pin a device. |
| `use_reciprocal_rank_fusion` | `False` | Use RRF instead of a cross-encoder. |
| `reciprocal_rank_fusion_constant` | `60.0` | The constant in the RRF score `1 / (rank + constant)`. |

The two re-rankers are mutually exclusive, and `DocChatAgent` resolves the combination for you:

- With `use_reciprocal_rank_fusion=True`, `cross_encoder_reranking_model` is ignored and a warning
  is logged. Set `use_reciprocal_rank_fusion=False` to use the cross-encoder.
- With `cross_encoder_reranking_model=""` (no `sentence-transformers`) **and**
  `use_reciprocal_rank_fusion=False`, several retrieval methods would leave the merged ranking
  undefined, so RRF is enabled automatically and the reason is logged. This only triggers when BM25
  or fuzzy matching is on and `n_relevant_chunks` is smaller than the number of candidates those
  methods contribute - which is the case in the default configuration, where
  `n_similar_chunks=3` per method feeds `n_relevant_chunks=3`.

```python
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig

config = DocChatAgentConfig(
    cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder_device="cuda",       # optional; None auto-selects
    use_reciprocal_rank_fusion=False,  # required for cross-encoder re-ranking
    n_similar_chunks=15,               # base candidate count, see below
    n_relevant_chunks=5,               # chunks kept after re-ranking
)
agent = DocChatAgent(config)
```

Re-ranking only pays off with a candidate pool larger than the final answer, so it changes how much
is retrieved: when a cross-encoder or RRF is active, the internal `retrieval_multiple` becomes `3`
and each enabled retrieval method fetches `n_similar_chunks * 3` candidates (`n_similar_chunks` when
neither re-ranker is active). The example above therefore retrieves up to 45 candidates per method,
and the cross-encoder scores all of them before `n_relevant_chunks` are kept. That is a model call
per candidate, so raising `n_similar_chunks` costs latency and memory proportionally.

## What the re-ranker does

`rerank_with_cross_encoder` scores every retrieved passage with the model, squashes the raw scores
with a sigmoid so they fall in `(0, 1)`, and returns the passages in descending score order. It only
**re-orders** - it does not drop passages. Which chunks survive is decided afterwards, in
`get_relevant_chunks`, by `n_relevant_chunks`.

Diversity re-ranking (`rerank_diversity`, default `True`) and periphery re-ranking
(`rerank_periphery`, default `True`, to counter the "lost in the middle" effect) then run on that
already-sorted list, so they can adjust the final order further. Set them to `False` to keep the
cross-encoder order verbatim.

## Model loading and concurrency

A `CrossEncoder` is instantiated once per process for a given model and device and cached, and its
`predict` calls are serialized through a lock, so tasks that share them neither load the model
repeatedly nor call it concurrently. `cross_encoder_device=None` is resolved to the device that would
be selected automatically (`cuda`, then `mps`, then `cpu`) **before** the cache key is built, so on a
CUDA host `None` and `"cuda"` are the same cache entry rather than two, which also means they share
one lock.

## Related

- [Chunking](chunking.md) - how `n_similar_chunks` chunks are produced in the first place.
- `examples/docqa/` - end-to-end `DocChatAgent` setups you can adapt.
