# LLM Client Connection Pool Exhaustion Issue

## Problem Statement

When using Langroid in multi-agent systems where agents are created dynamically (e.g., one agent per data row), each agent creates its own LLM client instance (OpenAI, Groq, or Cerebras). This pattern leads to connection pool exhaustion, resulting in "too many open files" errors and degraded performance.

## Current Behavior

### Client Creation Flow
1. Each `ChatAgent` instantiates its own `OpenAIGPT` instance
2. Each `OpenAIGPT` instance creates new client objects:
   - For Groq models: Creates `Groq()` and `AsyncGroq()` clients
   - For Cerebras models: Creates `Cerebras()` and `AsyncCerebras()` clients  
   - For OpenAI/others: Creates `OpenAI()` and `AsyncOpenAI()` clients
3. These clients maintain their own connection pools via httpx

### Problem Scenario
```python
# Anti-pattern: Creating many agents
for row in data[:100]:  # 100 rows
    agent = ChatAgent(config)  # Creates new OpenAI client
    result = agent.run(row)    # Makes API calls
    # Agent goes out of scope but connections may linger
```

This creates 100 separate OpenAI clients, each with its own connection pool.

## Impact

1. **Resource Exhaustion**: Each client maintains open connections, leading to file descriptor limits
2. **Performance Degradation**: Connection establishment overhead for each new client
3. **Potential API Rate Limiting**: Multiple clients may trigger more aggressive rate limiting
4. **Memory Usage**: Each client instance consumes memory for connection pools

## Root Cause

The issue stems from:
1. Lack of client reuse across agent instances
2. Connection pools not being properly closed when agents are garbage collected
3. The anti-pattern of creating many short-lived agents instead of reusing agents

## Constraints

1. **API Compatibility**: Solution must not break existing Langroid API
2. **Configuration Flexibility**: Different agents may need different configurations (API keys, base URLs, timeouts)
3. **Thread Safety**: Clients must be safely shareable across multiple agents
4. **Async Support**: Must handle both sync and async client variants

## Critical Considerations

### 1. Configuration Variations
Different agents in the same system might require different client configurations:
- **Different API Keys**: Agent A might use one OpenAI key, Agent B another
- **Different Base URLs**: Some agents might use standard OpenAI, others might use Azure OpenAI
- **Different Timeouts**: Long-running tasks might need higher timeouts
- **Different Headers**: Custom headers for different use cases

**Implication**: We cannot have just one singleton per client type. We need to cache clients based on their full configuration, creating a new client only when a unique configuration is encountered.

### 2. Thread Safety
Multiple agents might run concurrently and share the same client instance:
- The httpx library (used by OpenAI, Groq, Cerebras clients) is designed to be thread-safe
- Connection pools in httpx can handle concurrent requests
- No additional locking should be needed for client access

**Implication**: Shared clients can be used safely across multiple threads/agents without synchronization overhead.

### 3. Lifecycle Management
Proper cleanup of singleton clients is crucial:
- **When to close**: Clients hold network resources that should be released
- **Garbage collection**: Need to ensure clients can be GC'd when no longer needed
- **Application shutdown**: Should close all clients gracefully on exit

**Implications**: 
- Consider using weak references to allow garbage collection of unused clients
- Implement `atexit` hooks for graceful shutdown
- May need a manual cleanup mechanism for long-running applications
- Monitor for memory leaks from accumulating cached clients with unique configs

## Proposed Solution: Client Singleton Pattern

### Approach
Implement a caching layer that returns singleton clients based on configuration:

1. **Wrapper Functions**: Replace direct client instantiation with wrapper functions:
   - `get_openai_client(config) -> OpenAI`
   - `get_groq_client(config) -> Groq`
   - `get_cerebras_client(config) -> Cerebras`
   - Similar for async variants

2. **Configuration-Based Caching**: Cache clients keyed by their configuration parameters:
   - API key
   - Base URL
   - Timeout
   - Headers
   - Organization (for OpenAI)

3. **Implementation Location**: In `langroid/language_models/openai_gpt.py`, replace:
   ```python
   # Current
   self.client = OpenAI(api_key=self.api_key, ...)
   
   # Proposed
   self.client = get_openai_client(api_key=self.api_key, ...)
   ```

### Benefits
- Reduces client instances from N (number of agents) to M (unique configurations)
- No API changes required
- Follows OpenAI best practices for client reuse
- Transparent to existing code

### Alternative Solutions Considered

1. **Agent Pooling**: Reuse agents instead of creating new ones
   - Pros: Most efficient
   - Cons: Requires significant API changes

2. **Explicit Client Registry**: Pass shared clients to agents
   - Pros: Explicit control
   - Cons: Breaks existing API, requires user awareness

3. **Connection Limit Configuration**: Reduce connection pool sizes
   - Pros: Simple
   - Cons: Doesn't address root cause, may hurt performance

## Success Criteria

1. Creating 100+ agents should not cause "too many open files" errors
2. Memory usage should remain stable with many agents
3. No breaking changes to existing Langroid API
4. Performance improvement for multi-agent scenarios

## Implementation Notes

- httpx clients (used by OpenAI/Groq/Cerebras) are thread-safe
- Consider using weak references to allow garbage collection
- May need cleanup hooks (atexit) for proper shutdown
- Should add logging for cache hits/misses for debugging

## References

- OpenAI Cookbook: Best practices recommend reusing client instances
- httpx documentation: Connection pooling behavior
- Python file descriptor limits and ulimit settings