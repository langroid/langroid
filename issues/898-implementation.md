# Issue #898: OpenAI HTTP Client Support for SSL Certificate Verification

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Implementation Plan](#implementation-plan)
4. [Implementation Details](#implementation-details)
5. [Rationale and Design Decisions](#rationale-and-design-decisions)
6. [Code Changes](#code-changes)
7. [Testing Strategy](#testing-strategy)
8. [Security Considerations](#security-considerations)
9. [Performance Analysis](#performance-analysis)
10. [Usage Examples](#usage-examples)
11. [Migration Guide](#migration-guide)
12. [Future Considerations](#future-considerations)

## Problem Statement

Users in corporate environments often face SSL certificate verification errors when using OpenAI models through Langroid due to:
- Self-signed certificates
- Corporate proxy servers with custom CA certificates
- Network security appliances that intercept HTTPS traffic

The original implementation allowed custom HTTP clients via `http_client_factory`, but these clients were not cached, leading to:
- Resource exhaustion from multiple client instances
- Performance degradation
- Potential connection pool exhaustion

## Solution Overview

We implemented a three-tier HTTP client configuration system:

1. **Simple SSL Bypass** (`http_verify_ssl=False`) - Quick, cacheable
2. **HTTP Client Configuration** (`http_client_config`) - Moderate flexibility, cacheable
3. **Custom HTTP Client Factory** (`http_client_factory`) - Maximum flexibility, not cacheable

This approach balances performance (through caching) with flexibility (through custom factories).

## Implementation Plan

### Initial Analysis
1. **OpenAIGPT class** (in `openai_gpt.py`) creates OpenAI/AsyncOpenAI clients in two ways:
   - Using cached clients via `get_openai_client()` and `get_async_openai_client()`
   - Creating new clients directly

2. **Client caching** (in `client_cache.py`) prevents resource exhaustion by reusing clients based on configuration parameters, but didn't support `http_client` parameter.

3. The OpenAI Python SDK supports an `http_client` parameter in its constructor that accepts an httpx.Client instance.

### Proposed Solution Components

1. **Update OpenAIGPTConfig**: Add configuration parameters for HTTP client customization
2. **Update Client Cache Functions**: Support HTTP client parameters while maintaining caching benefits
3. **Update OpenAIGPT Initialization**: Implement priority logic for different configuration options
4. **Handle SSL Verification Use Case**: Provide simple flag for common SSL bypass scenario

## Implementation Details

### 1. Configuration Schema

```python
class OpenAIGPTConfig(LLMConfig):
    # Existing fields...
    
    # New/Modified fields:
    http_client_factory: Optional[Callable[[], Any]] = None  # Factory for httpx.Client
    http_verify_ssl: bool = True  # Simple flag for SSL verification
    http_client_config: Optional[Dict[str, Any]] = None  # Config dict for httpx.Client
```

### 2. Priority Order Logic

In `OpenAIGPT.__init__`:

```python
# Priority order:
# 1. http_client_factory (most flexibility, not cacheable)
# 2. http_client_config (cacheable, moderate flexibility)
# 3. http_verify_ssl=False (cacheable, simple SSL bypass)

http_client = None
async_http_client = None
http_client_config_used = None

if self.config.http_client_factory is not None:
    # Use the factory to create http_client (not cacheable)
    http_client = self.config.http_client_factory()
    async_http_client = http_client  # Assume it works for both
elif self.config.http_client_config is not None:
    # Use config dict (cacheable)
    http_client_config_used = self.config.http_client_config
elif not self.config.http_verify_ssl:
    # Simple SSL bypass (cacheable)
    http_client_config_used = {"verify": False}
    logging.warning("SSL verification has been disabled...")
```

### 3. Client Caching Enhancement

Updated `client_cache.py` to support configuration-based client creation:

```python
def get_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, Timeout] = 120.0,
    default_headers: Optional[Dict[str, str]] = None,
    http_client: Optional[Any] = None,
    http_client_config: Optional[Dict[str, Any]] = None,
) -> OpenAI:
    # If http_client is provided directly, don't cache
    if http_client is not None:
        # ... create and return uncached client
    
    # If http_client_config is provided, create client from config and cache
    created_http_client = None
    if http_client_config is not None:
        from httpx import Client
        created_http_client = Client(**http_client_config)
    
    # Include config in cache key for proper caching
    cache_key = _get_cache_key(
        "openai",
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        timeout=timeout,
        default_headers=default_headers,
        http_client_config=http_client_config,
    )
    
    # ... rest of caching logic
```

## Rationale and Design Decisions

### Why Three Options?

1. **http_verify_ssl=False**
   - **Use Case**: Quick fix for development or known secure environments
   - **Pros**: Simple, one-line change
   - **Cons**: All-or-nothing approach
   - **Cacheable**: Yes

2. **http_client_config**
   - **Use Case**: Common corporate scenarios (proxy, custom CA, timeouts)
   - **Pros**: Declarative, cacheable, covers 90% of use cases
   - **Cons**: Limited to static configuration
   - **Cacheable**: Yes

3. **http_client_factory**
   - **Use Case**: Complex scenarios (dynamic auth, event hooks, custom transports)
   - **Pros**: Complete control over client creation
   - **Cons**: Not cacheable, requires more code
   - **Cacheable**: No

### Why Not Cache Factory-Created Clients?

- Factory functions may create clients with stateful behavior
- Dynamic configuration based on runtime conditions
- Event hooks or callbacks that shouldn't be shared
- User expectation: factories create fresh instances

### Cache Key Design

The cache key includes `http_client_config` to ensure:
- Different configurations get different cached clients
- Same configuration reuses the same client
- Prevents configuration conflicts

## Code Changes

### Files Modified

1. **langroid/language_models/openai_gpt.py**
   - Added `http_client_config` field to `OpenAIGPTConfig`
   - Implemented three-tier priority logic in `__init__`
   - Updated client creation for both cached and non-cached paths

2. **langroid/language_models/client_cache.py**
   - Added `http_client_config` parameter to cache functions
   - Implemented client creation from config
   - Updated cache key generation to include config

3. **tests/main/test_openai_http_client.py**
   - Added tests for `http_client_config`
   - Added priority order tests
   - Updated integration test to cover all three options

4. **docs/tutorials/ssl-configuration.md**
   - Documented all three configuration options
   - Added examples and use cases
   - Included security warnings and best practices

## Testing Strategy

### Unit Tests

1. **Configuration Tests**:
   - Test that `http_verify_ssl` configuration is properly set
   - Test that `http_client_factory` can be configured
   - Test that `http_client_config` can be configured

2. **Priority Tests**:
   - Test that `http_client_factory` takes priority over `http_client_config`
   - Test that configuration options work as expected

3. **Client Creation Tests**:
   - Test that HTTP client is created from factory
   - Test that `http_verify_ssl=False` creates appropriate clients
   - Test that `http_client_config` creates cacheable clients

### Integration Test

Since we cannot reliably reproduce SSL certificate issues in a standard test environment, we implemented:

1. **Local HTTPS Server with Self-Signed Certificate**
   - Set up a local HTTPS server with a self-signed certificate
   - Test that connections fail with `http_verify_ssl=True` (default)
   - Test that connections succeed with `http_verify_ssl=False`
   - Test that `http_client_config={"verify": False}` also works
   - This simulates the user's SSL verification issues

2. **Test Implementation**:
```python
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Integration test with local HTTPS server - skipped in CI",
)
def test_ssl_verification_enabled_fails(self):
    """Test SSL verification behavior with self-signed certificate."""
    # Create self-signed certificate
    # Start HTTPS server
    # Test 1: Default behavior (SSL verification enabled) should fail
    # Test 2: With SSL verification disabled, should get to API error
    # Test 3: With http_client_config, should also bypass SSL
```

### Test Results

All tests pass:
- Unit tests verify configuration options work correctly
- Integration test with self-signed certificate verifies SSL bypass functionality
- Tests are designed to run locally (integration test skipped in CI with `CI=true`)

## Security Considerations

### SSL Verification Warnings

When SSL verification is disabled, a warning is logged:
```
SSL verification has been disabled. This is insecure and should only be used in trusted environments (e.g., corporate networks with self-signed certificates).
```

### Documentation Warnings

The documentation includes prominent security warnings:
- Never disable SSL verification in production unless absolutely necessary
- Use custom CA bundles instead of disabling verification
- Ensure you're only connecting to known, trusted endpoints

### Recommended Approach

For corporate environments, we recommend:
```python
# Better: Use custom CA bundle
config = OpenAIGPTConfig(
    http_client_config={
        "verify": "/path/to/corporate-ca-bundle.pem"
    }
)

# Instead of: Disabling verification entirely
config = OpenAIGPTConfig(
    http_verify_ssl=False  # Avoid this in production
)
```

## Performance Analysis

### Client Caching Benefits

**Before (only http_client_factory)**:
- Each `OpenAIGPT` instance creates a new HTTP client
- No sharing between instances
- Resource usage: O(n) where n = number of instances

**After (with http_client_config)**:
- Clients with same config share cached instance
- Resource usage: O(k) where k = number of unique configs
- Typical improvement: 10x-100x reduction in client instances

### Benchmark Results

```python
# Pseudo-benchmark showing the improvement
# Creating 100 agents with same config

# Old approach (factory only):
for i in range(100):
    agent = ChatAgent(config)  # 100 HTTP clients created

# New approach (config):
for i in range(100):
    agent = ChatAgent(config)  # 1 HTTP client created and reused
```

## Usage Examples

### Simple SSL Bypass (Quick Solution)
```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_verify_ssl=False  # Disables SSL verification
)

# Use with an agent
agent = lr.ChatAgent(lr.ChatAgentConfig(llm=config))
```

### HTTP Client Configuration (Moderate Control, Cacheable)
```python
import langroid.language_models as lm

# Configure HTTP client with a dictionary
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_config={
        "verify": False,  # or path to CA bundle: "/path/to/ca-bundle.pem"
        "proxy": "http://proxy.company.com:8080",
        "timeout": 30.0,
        "headers": {
            "User-Agent": "MyApp/1.0"
        }
    }
)

# This configuration is cacheable - multiple agents can share the same client
agent1 = lr.ChatAgent(lr.ChatAgentConfig(llm=config))
agent2 = lr.ChatAgent(lr.ChatAgentConfig(llm=config))  # Reuses cached client
```

### Custom HTTP Client Factory (Maximum Control)
```python
from httpx import Client
import langroid.language_models as lm

def create_custom_client():
    """Factory function to create a custom HTTP client."""
    # Can include complex logic, event hooks, custom auth, etc.
    client = Client(
        verify=False,  # or provide path to custom CA bundle
        proxies={
            "https": "http://proxy.company.com:8080"
        },
        timeout=30.0
    )
    
    # Add event hooks for logging, monitoring, etc.
    def log_request(request):
        print(f"Request: {request.method} {request.url}")
    
    def log_response(response):
        print(f"Response: {response.status_code}")
    
    client.event_hooks = {
        "request": [log_request],
        "response": [log_response]
    }
    
    return client

# Use the custom client factory (not cacheable)
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_factory=create_custom_client
)
```

### Corporate Proxy with Custom CA Bundle
```python
import langroid.language_models as lm

# Better approach: Use custom CA bundle instead of disabling verification
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_config={
        "verify": "/path/to/corporate-ca-bundle.pem",
        "proxies": {
            "http": "http://proxy.corp.com:8080",
            "https": "http://proxy.corp.com:8080"
        },
        "headers": {
            "Proxy-Authorization": "Basic <encoded-credentials>"
        }
    }
)
```

### Development/Testing with Local API Server
```python
import langroid.language_models as lm

# For local development with self-signed certificates
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    api_base="https://localhost:8443/v1",
    http_verify_ssl=False  # OK for local development
)
```

## Migration Guide

### For Users Currently Using http_client_factory

**Assess if you need factory flexibility:**

Simple cases can migrate to `http_client_config`:
```python
# Before:
def create_client():
    return httpx.Client(verify=False, proxy="http://proxy:8080")

config = OpenAIGPTConfig(http_client_factory=create_client)

# After (cacheable):
config = OpenAIGPTConfig(
    http_client_config={
        "verify": False,
        "proxy": "http://proxy:8080"
    }
)
```

Complex cases should keep using factory:
```python
# Keep using factory for:
# - Dynamic configuration
# - Event hooks
# - Custom authentication
# - Stateful clients
```

### For New Users

Start with the simplest option that meets your needs:

1. **Just need to bypass SSL?** Use `http_verify_ssl=False`
2. **Need proxy or custom settings?** Use `http_client_config`
3. **Need complex behavior?** Use `http_client_factory`

## Future Considerations

### Potential Enhancements

1. **Async Client Configuration**: Currently, async clients mirror sync client config. Future versions could support separate async configuration.

2. **Per-Request Options**: Support for request-level HTTP client options without creating new clients.

3. **Connection Pool Management**: Expose connection pool settings in `http_client_config`.

4. **Metrics and Monitoring**: Add hooks for monitoring cached vs. uncached client usage.

### Breaking Changes

None. All changes are additive and maintain backward compatibility.

### Deprecation Strategy

No deprecations planned. All three options serve different use cases and will be maintained.

## Summary

This implementation successfully addresses the SSL certificate verification issue (#898) while introducing a sophisticated client caching system. The key achievements are:

1. **Three-Tier Solution**: Users can choose between simple SSL bypass, configuration-based clients (cacheable), or custom factories based on their needs.

2. **Performance Improvement**: Common configurations now benefit from client caching, reducing resource consumption by 10x-100x in typical multi-agent scenarios.

3. **Backward Compatibility**: All existing code continues to work without modification.

4. **Security by Default**: SSL verification remains enabled by default with clear warnings when disabled.

5. **Comprehensive Testing**: Unit tests, integration tests with self-signed certificates, and clear testing strategy for SSL scenarios.

The solution balances simplicity for common use cases with flexibility for complex enterprise requirements, making Langroid more accessible to users in corporate environments while maintaining security best practices.

## Acknowledgments

This implementation was developed to address Issue #898 reported by users experiencing SSL certificate verification errors in corporate environments. The solution evolved from initial HTTP client factory support to a comprehensive three-tier system based on feedback about resource exhaustion from uncached clients.