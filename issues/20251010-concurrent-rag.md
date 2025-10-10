# Concurrent DocChatAgent Task Execution Investigation

**Date:** 2025-10-10
**Status:** In Progress - Root cause partially identified, fix implemented but incomplete
**Priority:** Medium

## Problem Statement

When running multiple DocChatAgent tasks concurrently using Langroid's `run_batch_tasks` mechanism, the tasks execute **sequentially** rather than concurrently, despite being submitted to `asyncio.gather`. This results in no performance improvement over sequential execution.

### Expected Behavior
- 10 queries submitted concurrently should execute in parallel
- Total time should be approximately equal to the slowest single query
- Logs should show overlapping VecDB searches and LLM calls

### Actual Behavior
- All tasks are submitted together (START logs show same timestamp)
- But they execute one after another with 4-5 second gaps between VecDB searches
- Total time: ~60 seconds for 10 queries (~6 sec/query) - same as sequential
- VecDB search logs show sequential pattern:
  - Query 1: 11:56:21
  - Query 2: 11:56:25 (4 sec later)
  - Query 3: 11:56:30 (5 sec later)
  - Query 4: 11:56:35 (5 sec later)

## Investigation & Findings

### Scripts Created

#### 1. `examples/docqa/rag-concurrent.py`
Main test script with timing instrumentation:
- Loads Borges' "Library of Babel" story from URL
- Runs 10 predefined questions about the story
- Supports `--sequential` flag for true sequential baseline (simple loop)
- Default mode uses `run_batch_tasks(..., sequential=False)` for asyncio concurrency
- Includes timestamp logging (HH:MM:SS.mmm) and thread ID tracking
- **Usage:**
  ```bash
  # Concurrent mode (default)
  python3 examples/docqa/rag-concurrent.py

  # True sequential baseline
  python3 examples/docqa/rag-concurrent.py --sequential
  ```

#### 2. `examples/docqa/rag-concurrent-debug.py`
Debug version with additional instrumentation:
- Subclasses `DocChatAgent` to add timing to vecstore and LLM operations
- Fewer questions (3 by default) for faster testing
- Tracks phase-level timing (VECSTORE_START/END, LLM_CALL_START/END)
- **Usage:**
  ```bash
  python3 examples/docqa/rag-concurrent-debug.py --num_questions=3
  ```

#### 3. `test_vecdb_clone.py`
Minimal test to verify vecdb cloning behavior:
- Creates a DocChatAgent with vecdb
- Clones task 3 times
- Prints vecdb and client IDs to verify separate instances
- **Usage:**
  ```bash
  python3 test_vecdb_clone.py
  ```

### Root Cause Analysis

#### Initial Hypothesis: Shared VecDB Instance
**Code Location:** `langroid/agent/chat_agent.py:283` (line number before fix)

Original clone() method did shallow copy:
```python
# OLD CODE - PROBLEM
new_agent.vecdb = self.vecdb  # All clones share same vecdb!
```

**Impact:**
- All cloned agents shared a single QdrantDB client instance
- For local storage: `.lock` file prevents concurrent access
- For cloud storage: shared connection pool may limit concurrency

#### Fix Implemented
**Code Location:** `langroid/agent/chat_agent.py:282-310`

```python
# NEW CODE - PARTIAL FIX
if self.vecdb is not None:
    from langroid.vector_store.base import VectorStore
    from langroid.vector_store.qdrantdb import QdrantDBConfig
    import os

    vecdb_config = self.vecdb.config.model_copy(deep=True)
    vecdb_config.replace_collection = False

    # For local Qdrant storage, use separate paths for each clone
    if isinstance(vecdb_config, QdrantDBConfig) and not vecdb_config.cloud:
        base_path = vecdb_config.storage_path.rstrip('/')
        clone_path = f"{base_path}_clone_{i}_{os.getpid()}"
        vecdb_config.storage_path = clone_path
        # Copy collection data to clone path
        if not os.path.exists(clone_path):
            import shutil
            if os.path.exists(base_path):
                shutil.copytree(base_path, clone_path)

    new_agent.vecdb = VectorStore.create(vecdb_config)
```

**What This Fixes:**
- ✅ Each clone gets its own vecdb instance and client
- ✅ For local storage: separate `.qdrant_clone_*` directories avoid `.lock` contention
- ✅ Verified working: `test_vecdb_clone.py` shows different IDs for each clone

**What's Still Broken:**
- ❌ Tasks still execute sequentially even with separate clients
- ❌ The fix helps for local storage but doesn't solve cloud Qdrant serialization

### Outstanding Issues

#### Issue #1: Cloud Qdrant Serialization
**Discovery:** Test environment uses `cloud=True` (remote Qdrant server)
- Despite separate clients, execution is still sequential
- Possible causes:
  1. **Connection pool limits** on cloud Qdrant instance
  2. **Rate limiting** on the Qdrant API
  3. **Client-side connection manager** serializing requests
  4. **Async/await pattern** in Qdrant client may not be truly concurrent

**Evidence:**
```
🔧 Clone 0: vecdb type=QdrantDBConfig, cloud=True
🔧 Clone 1: vecdb type=QdrantDBConfig, cloud=True
🔧 Clone 2: vecdb type=QdrantDBConfig, cloud=True
```

#### Issue #2: Logging Artifacts
The START/COMPLETE logging in `rag-concurrent.py` is misleading:
- `input_map` logs START events (happens during task submission)
- `output_map` logs COMPLETE events (called AFTER all tasks finish)
- All COMPLETE logs show same timestamp because they're logged in a batch

**Better indicators:**
- VecDB INFO logs show actual execution timing
- Look for "Searching VecDB" timestamps in the logs

## Next Steps

### Immediate Actions Needed

1. **Investigate Qdrant Client Concurrency**
   - File: `langroid/vector_store/qdrantdb.py`
   - Check if `QdrantClient.search_batch()` is truly async
   - Look for any locks, semaphores, or connection pooling
   - Test: Create minimal script with just Qdrant queries (no DocChatAgent)

2. **Test with Local Qdrant**
   - Verify the clone-path fix works for local storage
   - Set `cloud=False` in DocChatAgentConfig
   - Confirm separate `.qdrant_clone_*` directories are created
   - Verify no `.lock` contention

3. **Profile Async Execution**
   - Add more granular timing inside `run_batch_tasks`
   - Check if `asyncio.gather` is actually creating concurrent tasks
   - Use Python's `asyncio` debug mode
   - Instrument the actual async chain: input_map → clone → run_async → output_map

4. **Check LLM Client**
   - Although not primary suspect, verify LLM client is concurrent
   - File: `langroid/language_models/openai_gpt.py`
   - Check for shared HTTP clients or connection pools

### Testing Strategy

```bash
# 1. Test local storage fix
python3 examples/docqa/rag-concurrent.py --sequential  # Baseline
python3 examples/docqa/rag-concurrent.py               # With fix
# Check for .qdrant_clone_* directories and timing improvement

# 2. Test cloud Qdrant concurrency
# (Current failing case)
# Need to identify why separate clients don't help

# 3. Minimal Qdrant concurrency test
# Create standalone script that just does concurrent Qdrant searches
# Isolate the vecdb layer from DocChatAgent complexity
```

### Files Modified

1. `langroid/agent/chat_agent.py` - clone() method (lines 282-312)
2. `examples/docqa/rag-concurrent.py` - NEW test script
3. `examples/docqa/rag-concurrent-debug.py` - NEW debug script
4. `test_vecdb_clone.py` - NEW minimal test (root directory)

### Files to Investigate Next

1. `langroid/vector_store/qdrantdb.py` - QdrantDB client implementation
2. `langroid/agent/batch.py` - Batch task execution (already reviewed)
3. `langroid/language_models/openai_gpt.py` - LLM client concurrency
4. Qdrant client library itself - may need to check external dependency

## Context for Next Agent

### What Works
- ✅ Vecdb cloning creates separate instances
- ✅ Test scripts are functional and provide good visibility
- ✅ Can see the sequential execution pattern in logs

### What Doesn't Work
- ❌ Concurrent execution still runs sequentially
- ❌ ~60 seconds for 10 queries (no improvement over sequential)
- ❌ Root cause for cloud Qdrant serialization unknown

### Key Insight
The problem is **NOT just the shallow vecdb copy**. That's fixed. The deeper issue is that even with separate QdrantDB clients accessing a cloud instance, the queries serialize. This suggests:
- A shared resource deeper in the stack (connection pool, rate limiter, etc.)
- Or the async pattern isn't actually concurrent
- Or Qdrant cloud itself is serializing (unlikely but possible)

### Recommended Approach
1. Create minimal reproduction with just Qdrant (no DocChatAgent)
2. Enable asyncio debug mode to see task scheduling
3. Profile with `asyncio` task timing
4. Consider testing with a different vector store (Chroma, etc.) to isolate Qdrant

## References

- Original issue report: User experiencing sequential execution when expecting concurrent
- Test document: "The Library of Babel" by Jorge Luis Borges
- URL: https://xpressenglish.com/our-stories/library-of-babel/
- Langroid batch task mechanism: `langroid/agent/batch.py`
- Qdrant lock file location: `.qdrant/doc-chat/.lock`

## Additional Notes

### Understanding `sequential` Parameter
The `sequential` parameter in `run_batch_tasks` does NOT mean "use a simple loop":
- `sequential=False`: Uses `asyncio.gather` for concurrent execution
- `sequential=True`: Uses `await` in a loop (still async, just not concurrent)

For TRUE sequential baseline, use a simple for-loop calling `task.run()` directly (as implemented in `rag-concurrent.py --sequential`).

### Logging Best Practices
- Don't trust START/COMPLETE logs from input_map/output_map
- Look at actual operation logs (VecDB searches, LLM calls)
- Use timestamps to verify overlapping execution windows
- Thread IDs help identify parallel execution
