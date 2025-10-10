# Concurrent DocChatAgent Batch Execution

**Date:** 2025-10-10  
**Status:** Resolved  
**Priority:** Medium

## Summary
Batch DocChatAgent runs submitted via `run_batch_tasks(..., sequential=False)` were completing one-by-one because `DocChatAgent.llm_response_async` awaited the fully synchronous `answer_from_docs`, blocking the event loop. Cloned tasks therefore serialized on retrieval/LLM work even though `asyncio.gather` was used.

## Fix
- Wrap `answer_from_docs` with `asyncio.to_thread` inside `DocChatAgent.llm_response_async`, letting each request execute on the default thread pool while the event loop schedules other tasks.
- Generalize vector-store cloning: `ChatAgent.clone()` now delegates to `vecdb.clone()`, the base `VectorStore` deep-copies config and instantiates a fresh store, and `QdrantDB.clone()` simply relies on the base behaviour to spin up independent clients for cloud deployments while keeping local instances shared for file-lock safety.
- Rework `examples/docqa/rag-concurrent.py` to drive task clones with `asyncio.as_completed`, capture per-question START/WORKER/COMPLETE events (including thread IDs and timings), add a `--log-only` mode plus filtering instructions for clean concurrency proof, and expose a `--use-builtin-batch` flag to exercise the original `run_batch_tasks` harness.
- Update the debug script to pass through `query_proxies`, keeping its instrumentation compatible with the main agent, and add a DocChat `run_batch_tasks` regression test covering multiple vector stores.

## Verification
- `uv run python examples/docqa/rag-concurrent.py --num-questions=3`
- `uv run python examples/docqa/rag-concurrent.py --sequential --num-questions=3`
- `uv run python examples/docqa/rag-concurrent.py --num-questions=3 --log-only`
- `uv run python examples/docqa/rag-concurrent.py --use-builtin-batch --num-questions=3 --log-only`
- `uv run python examples/docqa/rag-concurrent-debug.py --num_questions=3`

Concurrent runs now finish ~2× faster than the sequential baseline, the log summary shows overlapping worker threads, and the new regression test (`pytest tests/main/test_vector_stores.py::test_doc_chat_batch_with_vecdb_cloning[...]`) passes across supported vector stores, confirming both concurrency and cloned-store isolation.*** End Patch
