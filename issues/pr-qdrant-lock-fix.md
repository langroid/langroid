# Fix QdrantDB Lock File Issue

## Problem
When using QdrantDB with local storage, file lock conflicts occurred when:
1. A QdrantDB instance was created but not properly closed
2. Another part of the code tried to create a new QdrantDB instance with the same storage path
3. Qdrant would detect the `.lock` file and create a new storage directory (e.g., `./qdrant_data.new`)

## Solution
1. **Added `close()` method to QdrantDB** - Calls the underlying client's close method to release the file lock
2. **Added context manager support** - Implemented `__enter__` and `__exit__` for automatic cleanup
3. **Fixed DocChatAgent's `clear()` method** - Now closes the old vecdb before creating a new one

## Usage
```python
# Option 1: Explicit close
vecdb = QdrantDB(config)
vecdb.clear_all_collections(really=True)
vecdb.close()  # Release the lock

# Option 2: Context manager (automatic cleanup)
with QdrantDB(config) as vecdb:
    vecdb.clear_all_collections(really=True)
    # Automatically closed when exiting context
```

## Changes
- `langroid/vector_store/qdrantdb.py`: Added `close()`, `__enter__`, `__exit__` methods
- `langroid/agent/special/doc_chat_agent.py`: Fixed `clear()` to close old vecdb instance

This fix prevents the proliferation of `.new` directories when using QdrantDB with local storage.