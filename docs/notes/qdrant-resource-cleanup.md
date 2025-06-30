# QdrantDB Resource Cleanup

When using QdrantDB with local storage, it's important to properly release resources
to avoid file lock conflicts. QdrantDB uses a `.lock` file to prevent concurrent
access to the same storage directory.

## The Problem

Without proper cleanup, you may encounter this warning:

```
Error connecting to local QdrantDB at ./qdrant_data:
Storage folder ./qdrant_data is already accessed by another instance of Qdrant
client. If you require concurrent access, use Qdrant server instead.
Switching to ./qdrant_data.new
```

This happens when a QdrantDB instance isn't properly closed, leaving the lock file
in place.

## Solutions

### Method 1: Explicit `close()` Method

Always call `close()` when done with a QdrantDB instance:

```python
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

config = QdrantDBConfig(
    cloud=False,
    collection_name="my_collection",
    storage_path="./qdrant_data",
)

vecdb = QdrantDB(config)
# ... use the vector database ...
vecdb.clear_all_collections(really=True)

# Important: Release the lock
vecdb.close()
```

### Method 2: Context Manager (Recommended)

Use QdrantDB as a context manager for automatic cleanup:

```python
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

config = QdrantDBConfig(
    cloud=False,
    collection_name="my_collection", 
    storage_path="./qdrant_data",
)

with QdrantDB(config) as vecdb:
    # ... use the vector database ...
    vecdb.clear_all_collections(really=True)
    # Automatically closed when exiting the context
```

The context manager ensures cleanup even if an exception occurs.

## When This Matters

This is especially important in scenarios where:

1. You create temporary QdrantDB instances for maintenance (e.g., clearing
   collections)
2. Your application restarts frequently during development
3. Multiple parts of your code need to access the same storage path sequentially

## Note for Cloud Storage

This only affects local storage (`cloud=False`). When using Qdrant cloud service,
the lock file mechanism is not used.