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

<!--AGENT -- look at this new error:-->
## Update 2025-10-11: error involving cross-encoding re-ranker


tests/test_concurrent_rag_simple.py:193:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:354: in run_batch_task_gen
    return run_batched_tasks(
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:265: in run_batched_tasks
    return asyncio.run(run_all_batched_tasks(inputs, batch_size))
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/base_events.py:650: in run_until_complete
    return future.result()
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:231: in run_all_batched_tasks
    results = await _process_batch_async(
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:179: in _process_batch_async
    results = [handle_error(e) for _ in inputs]
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:179: in <listcomp>
    results = [handle_error(e) for _ in inputs]
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:102: in handle_error
    raise e
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:162: in _process_batch_async
    await asyncio.gather(
.venv/lib/python3.11/site-packages/langroid/agent/batch.py:330: in _do_task
    result = await task_i.run_async(
.venv/lib/python3.11/site-packages/langroid/agent/task.py:1020: in run_async
    await self.step_async()
.venv/lib/python3.11/site-packages/langroid/agent/task.py:1352: in step_async
    result = await self.response_async(r, turns)
.venv/lib/python3.11/site-packages/langroid/agent/task.py:1711: in response_async
    result = await response_fn(self.pending_message)
.venv/lib/python3.11/site-packages/langroid/agent/special/doc_chat_agent.py:864: in llm_response_async
    response = await asyncio.to_thread(self.answer_from_docs, query_str)
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/threads.py:25: in to_thread
    return await loop.run_in_executor(None, func_call)
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/futures.py:287: in __await__
    yield self  # This tells Task to wait for completion.
../../.pyenv/versions/3.11.0/lib/python3.11/asyncio/futures.py:203: in result
    raise self._exception.with_traceback(self._exception_tb)
../../.pyenv/versions/3.11.0/lib/python3.11/concurrent/futures/thread.py:58: in run
    result = self.fn(*self.args, **self.kwargs)
.venv/lib/python3.11/site-packages/langroid/agent/special/doc_chat_agent.py:1605: in answer_from_docs
    query, extracts = self.get_relevant_extracts(query)
.venv/lib/python3.11/site-packages/langroid/agent/special/doc_chat_agent.py:1495: in get_relevant_extracts
    passages = self.get_relevant_chunks(query, proxies)  # no LLM involved
.venv/lib/python3.11/site-packages/langroid/agent/special/doc_chat_agent.py:1433: in get_relevant_chunks
    passages = self.rerank_with_cross_encoder(query, passages)
.venv/lib/python3.11/site-packages/langroid/agent/special/doc_chat_agent.py:1115: in rerank_with_cross_encoder
    scores = model.predict([(query, p.content) for p in passages])
.venv/lib/python3.11/site-packages/sentence_transformers/cross_encoder/CrossEncoder.py:336: in predict
    self.model.to(self._target_device)
.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:4110: in to
    return super().to(*args, **kwargs)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:1355: in to
    return self._apply(convert)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:915: in _apply
    module._apply(fn)
.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:942: in _apply
    param_applied = fn(param)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

t = Parameter containing:
tensor(..., device='meta', size=(1536, 384), requires_grad=True)

    def convert(t):
        try:
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                    memory_format=convert_to_format,
                )
            return t.to(
                device,
                dtype if t.is_floating_point() or t.is_complex() else None,
                non_blocking,
            )
        except NotImplementedError as e:
            if str(e) == "Cannot copy out of meta tensor; no data!":
>               raise NotImplementedError(
                    f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                    f"when moving module from meta to a different device."
                ) from None
E               NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

.venv/lib/python3.11/site-packages/torch/nn/modules/module.py:1348: NotImplementedError
