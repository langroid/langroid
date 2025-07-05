# Release Notes - v0.56.11

## OpenAI Client Connection Management

### HTTP Client Caching
- Implements intelligent client caching for OpenAI and compatible APIs (Groq, Cerebras, etc.)
- Agents with identical configurations now share underlying HTTP clients
- Prevents "too many open files" errors when creating many agents (e.g., 100 agents for 100 data rows)
- Thread-safe implementation allows safe client sharing across threads

### Performance Improvements
- Reduced latency through connection reuse
- Eliminates redundant TCP handshakes
- Decreases CPU usage and network round-trips
- Leverages httpx's built-in connection pooling

### Configuration
- New `use_cached_client` parameter in `OpenAIGPTConfig` (enabled by default)
- Can be disabled for specific use cases:
  ```python
  config = OpenAIGPTConfig(
      chat_model="gpt-4",
      use_cached_client=False  # Disable caching
  )
  ```

### When to Disable Client Caching
- Multiprocessing environments (each process needs its own client)
- When complete client isolation is required between agents
- Debugging client-related issues
- Legacy code that depends on unique client instances

### Technical Implementation
- SHA256-based cache key generation for configuration uniqueness
- Singleton pattern with lazy initialization
- Automatic cleanup via atexit hooks
- Compatible with both sync and async OpenAI clients