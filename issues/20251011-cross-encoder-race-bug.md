# Cross-Encoder Reranker Race Condition Plan

## Summary

Concurrent DocChatAgent tasks that enable `cross_encoder_reranking_model`
raise a PyTorch `NotImplementedError` ("Cannot copy out of meta tensor; no data!")
intermittently. The failure originates inside `CrossEncoder.predict()` when the
underlying Hugging Face model is moved between devices while still in the meta
state. Multiple threads instantiating and using the same cross encoder at once
trigger this race.

## Current Reproduction Status

- `tests/main/test_concurrent_rag_simple.py` fails intermittently on `main` and
  on the working branch when run several times in a row (10–20 iterations).
- Failures occur only when `cross_encoder_reranking_model` is set and multiple
  tasks run concurrently; sequential runs pass.

## Root Cause Hypothesis

1. Each DocChatAgent clone instantiates its own `CrossEncoder` inside
   `rerank_with_cross_encoder()`.
2. SentenceTransformers lazily initializes the underlying HF model on the first
   call to `.predict()`. During initialization, `model.to(device)` tries to copy
   tensors out of the “meta” device.
3. When multiple clones call `.predict()` at the same time, they each try to
   load/transfer shared parameters simultaneously, and one thread encounters the
   `meta` tensor copy race, causing the `NotImplementedError`.

## Investigation Tasks

1. **Confirm shared-state behavior**
   - Inspect `CrossEncoder.predict` to verify it performs `self.model.to(...)`
     on each call, making it unsafe to invoke from multiple threads without
     coordination.
   - Capture concurrent stack traces/logs during failure to confirm multiple
     threads enter the to() conversion simultaneously.

2. **Reproduce in isolation**
   - Write a minimal script that spawns several threads; each thread loads the
     same cross-encoder model and immediately calls `.predict()` to reproduce
     the meta-tensor race outside Langroid. This will clarify whether the bug
     is entirely in PyTorch/HF or also in Langroid’s usage.

3. **Benchmark loading cost**
   - Measure time to instantiate `CrossEncoder` and to run `.predict()` so we
     understand the overhead when caching the model vs. reloading on demand.

## Proposed Fix

Implement a per-model cache with synchronization so each process holds one
`CrossEncoder` instance per model name:

1. **Global cache**
   - Introduce a module-level helper (e.g., `_get_cross_encoder(model_name)`) in
     `doc_chat_agent.py` that stores models in a dictionary keyed by
     `model_name`.
   - Guard cache creation with a global `threading.Lock` to avoid double
     instantiation.

2. **Per-model execution lock**
   - Associate each cached model with a reentrant `Lock`. Before calling
     `predict`, acquire the lock to serialize access. This prevents concurrent
     `.predict()` calls from moving the model between devices at the same time.

3. **Stable device assignment**
   - Force the cached model onto a specific device once (likely CPU unless
     configured otherwise). Skip repeated `model.to()` calls inside the lock so
     subsequent predictions reuse the initialized weights without touching the
     meta tensors.

4. **Agent changes**
   - Update `DocChatAgent.rerank_with_cross_encoder` to fetch the cached
     `(model, lock)` pair and run prediction inside the per-model lock.

## Validation Plan

1. Run `tests/main/test_concurrent_rag_simple.py` in a loop (e.g., 20 times) to
   ensure the race no longer triggers.
2. Execute the sequential control test and a small subset of the wider suite to
   confirm no regressions.
3. Optionally stress-test with more concurrent tasks and different
   `cross_encoder_reranking_model` values to ensure the cache handles multiple
   models correctly.

## Follow-Up Considerations

- Document the shared-model behavior near the config option so users know the
  reranker is serialized per model.
- Evaluate batching requests through the shared cross encoder in future work to
  regain some concurrency while avoiding race conditions.

## Progress Log

- **2025-10-11:** Implemented thread-safe cross-encoder cache in `DocChatAgent` to reuse a single model instance per name and serialize `.predict()` calls. Adjusted reranker to disable the default progress bar for batch runs.
- **2025-10-11:** Validated the fix by running `uv run pytest tests/main/test_concurrent_rag_simple.py -k cross_encoder -x` once and then in a 10× loop; all iterations passed without reproducing the meta-tensor error.
- **2025-10-11:** Defaulted cached cross encoders to CPU but added `cross_encoder_device` override on `DocChatAgentConfig` so users with GPUs can opt in while keeping library-safe defaults.
- **2025-10-11:** Added `--cross-encoder-device` pytest option (with optional `TEST_CROSS_ENCODER_DEVICE` env fallback) so the concurrency test can be run against CPU, CUDA, or MPS paths without code edits.
