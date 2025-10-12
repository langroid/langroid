# PR 926 Summary and Notes

## Pull Request Description

- fixed concurrent reranking by adding a shared cross-encoder cache (auto CUDA/MPS/CPU, optional override) and documenting the setup
- broadened `DocChatAgent` to accept any `LLMConfig`, cleaned up vector-store embedding cloning, and kept the concurrency demo relying on the default VecDB with opt-in flags for cross encoder/local embeddings
- expanded regression coverage (`tests/main/test_concurrent_rag_simple.py`) and updated docs for cross-encoder usage and device toggles

**Validation**
- `uv run pytest tests/main/test_concurrent_rag_simple.py -k cross_encoder -x`
- `uv run pytest -xvs tests/main/test_vector_stores.py::test_doc_chat_batch_with_vecdb_cloning`
- `uv run ruff check .`

## Cross-Encoder vs Embedding Model Handling

`DocChatAgent` relies on two model types when it runs multiple concurrent tasks:

1. **Embedding model** (part of the vector store) used for similarity retrieval. When clones shared the same embedding model instance, local SentenceTransformer-based models could clash. We now clone the embedding model per agent clone (lightweight enough to duplicate) so each clone gets a clean instance.
2. **Cross-encoder reranker** used to score passages jointly with the query. Duplication is expensive, so we cache a single instance per `(model, device)` and serialize `predict` calls behind a lock. This keeps GPU/CPU usage efficient while eliminating the "meta tensor" race.

In short: embeddings are cloned per clone for isolation; the cross encoder is shared but guarded for thread-safe access.
