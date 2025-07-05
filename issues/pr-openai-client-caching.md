# OpenAI Client Connection Management

## Problem
Creating many agents (e.g., 100 agents for 100 data rows) leads to "too many open files" errors due to each agent creating its own HTTP client, exhausting file descriptors.

## Solution
Implemented client caching/singleton pattern to reuse HTTP clients across multiple agent instances with the same configuration.

## Changes

### 1. Client Caching Module (`langroid/language_models/client_cache.py`)
- Singleton pattern for HTTP client reuse
- SHA256-based cache keys for configuration
- Wrapper functions for each client type (OpenAI, Groq, Cerebras)
- Lifecycle management with `atexit` hooks

### 2. OpenAIGPT Integration
- Added `use_cached_client: bool = True` config parameter
- Updated client creation to use wrapper functions when caching enabled
- Allows disabling for testing/special cases

### 3. ChatAgent Cleanup
- Updated `__del__` method to avoid closing shared clients
- Clients now managed centrally via client_cache module

### 4. Comprehensive Tests
- Tests for singleton behavior across all client types
- Verification of concurrent async usage
- Tests for model prefix routing (groq/, cerebras/, etc.)
- Regression tests with `use_cached_client` flag

## Benefits
- Prevents resource exhaustion when creating many agents
- Improves performance through connection pooling
- Backward compatible with opt-out capability
- Thread-safe for concurrent usage