# Concurrent DocChat RAG – Current Status (2025-10-10)

## Summary
- Sequential DocChat queries work against both cloud and local (Docker) Qdrant backends.
- Concurrent DocChat via `run_batch_tasks` now returns full answers instead of `DO-NOT-KNOW`; `examples/docqa/rag-concurrent.py --local-embeddings --use-builtin-batch` shows 900 char responses in concurrent mode.
- Key fixes in place:
  - `EmbeddingModel.clone()` + `VectorStore.clone()` ensure each clone gets an independent embedding model and leaves `replace_collection=False`.
  - `ChatAgent.clone()` delegates to `_clone_extra_state`, with `DocChatAgent` copying `chunked_docs` and related caches.
  - `DocChatAgent.get_relevant_extracts` now falls back to in-memory `chunked_docs` when the vector store collection is missing/empty, preventing premature `DO-NOT-KNOW`.
  - Regression test `tests/main/test_concurrent_doc_chat_qdrant.py` passes on the fix branch (uses real Qdrant + SentenceTransformer embeddings + MockLM) and fails on main after we drop the backing collection to force the fallback path.

## Findings
1. **Guardrail gap** – The original `get_relevant_extracts` short-circuited whenever Qdrant reported `points_count=0`, even if `chunked_docs` were populated. Clones hit this path because a fresh client often reports zero points immediately after ingest. The fallback resolves this by using the cached chunks whenever the vector store hasn’t caught up yet.
2. **Regression coverage** – The updated pytest harness no longer monkeypatches retrieval. It exercises the full `run_batch_tasks` flow against local Qdrant, with a `MockLM` to avoid external API calls. On main it fails at the `clone.chunked_docs` assertion, confirming the test’s regression behavior.
3. **Example validation** – Running the concurrent example with `--use-builtin-batch` and `--local-embeddings` now yields overlapping worker logs and long-form answers; deleting the collection post-ingest reproduces the legacy failure on main but passes with the new fallback.

## Outstanding Items
- Ensure CI spins up Qdrant before running `tests/main/test_concurrent_doc_chat_qdrant.py` (workflow already starts the container; keep an eye on readiness timing).
- Monitor for any cases where both vector store and `chunked_docs` are empty (e.g., ingest skipped). The new fallback will still produce `DO-NOT-KNOW` in that scenario, which is expected.
- Verify cloud Qdrant regression: run the concurrent example against a remote collection to ensure the fallback doesn’t mask real empty collections.

## Next Steps
1. Add a short CI check (or doc note) to confirm Qdrant health before pytest kicks off.
2. Evaluate whether we should log a debug message when the fallback path is used—helpful for diagnosing future data-sync delays.
3. Consider extending regression coverage to include the cloud Qdrant path once a stable test fixture exists.

## Fix Timeline (2025-10-08 → 2025-10-11)
- **Async blocking in DocChatAgent** (see `issues/20251010-concurrent-rag.md`): `llm_response_async` waited on synchronous retrieval, so `asyncio.gather` serialized every task. We wrapped `answer_from_docs` with `asyncio.to_thread`, letting concurrent tasks progress while the main event loop stays free.
- **Clone safety & retrieval fallback** (see `issues/20251010-concurrent-rag-codex.md`): cloned agents were reusing embedding models and losing access to cached chunks when Qdrant reported zero points. We taught embedding/vector-store configs to clone themselves and had `get_relevant_extracts` fall back to in-memory `chunked_docs`, restoring parallel runs with local embeddings.
- **Cross-encoder race condition** (see `issues/20251011-cross-encoder-race-bug.md`): simultaneous reranker calls tried to move a shared `CrossEncoder` between devices, triggering the PyTorch “meta tensor” error. A per-model cache plus locking (defaulting to CPU, override via `cross_encoder_device`) now keeps concurrent reranks deterministic across CPU, CUDA, and MPS.
