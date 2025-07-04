# OpenAI Client Caching

## Overview

Langroid implements client caching for OpenAI and compatible APIs (Groq, Cerebras, etc.) to improve performance and prevent resource exhaustion issues.

## Configuration

### Option
Set `use_cached_client` in your `OpenAIGPTConfig`:

```python
from langroid.language_models import OpenAIGPTConfig

config = OpenAIGPTConfig(
    chat_model="gpt-4",
    use_cached_client=True  # Default
)
```

### Default Behavior
- `use_cached_client=True` (enabled by default)
- Clients with identical configurations share the same underlying HTTP connection pool
- Different configurations (API key, base URL, headers, etc.) get separate client instances

## Benefits

- **Connection Pooling**: Reuses TCP connections, reducing latency and overhead
- **Resource Efficiency**: Prevents "too many open files" errors when creating many agents
- **Performance**: Eliminates connection handshake overhead on subsequent requests
- **Thread Safety**: Shared clients are safe to use across threads

## When to Disable Client Caching

Set `use_cached_client=False` in these scenarios:

1. **Multiprocessing**: Each process should have its own client instance
2. **Client Isolation**: When you need complete isolation between different agent instances
3. **Debugging**: To rule out client sharing as a source of issues
4. **Legacy Compatibility**: If your existing code depends on unique client instances

## Example: Disabling Client Caching

```python
config = OpenAIGPTConfig(
    chat_model="gpt-4",
    use_cached_client=False  # Each instance gets its own client
)
```

## Technical Details

- Uses SHA256-based cache keys to identify unique configurations
- Implements singleton pattern with lazy initialization
- Automatically cleans up clients on program exit via atexit hooks
- Compatible with both sync and async OpenAI clients