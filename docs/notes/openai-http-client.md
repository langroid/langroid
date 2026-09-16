# OpenAI HTTP Client Configuration

When using OpenAI models through Langroid in corporate environments or behind proxies, you may encounter SSL certificate verification errors. Langroid provides three flexible options to configure the HTTP client used for OpenAI API calls.

## Which `httpx` family to use (openai 2.x vs 3.x)

Langroid supports both `openai` 2.x and 3.x. The 2.x SDK is built on
[`httpx`](https://www.python-httpx.org/); the 3.x SDK replaced it with
[`httpx2`](https://httpx2.pydantic.dev/), the Pydantic-maintained successor with
the same `Client` / `AsyncClient` / `Timeout` API. Each SDK major accepts HTTP
clients from its own family, so when you build a client yourself (the factory
option below, or the `httpx.Auth` recipe in
[Rotating API Keys](rotating-api-keys.md)), import it from the family that
matches your installed SDK. `langroid.language_models.httpx_compat` does this
for you:

```python
from langroid.language_models.httpx_compat import import_httpx

httpx = import_httpx()  # the `httpx` module on openai 2.x, `httpx2` on 3.x
client = httpx.Client(verify=False)
```

The `http_verify_ssl` and `http_client_config` options need no change: langroid
constructs those clients through the same shim. The examples below use `httpx`
imports for brevity; substitute `httpx2` (or the shim) on openai 3.x.

Note: `litellm` still pins `openai<3`, so installing the `litellm` extra keeps
you on openai 2.x until that pin is lifted.

## Configuration Options

### 1. Simple SSL Verification Bypass

The quickest solution for development or trusted environments:

```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_verify_ssl=False  # Disables SSL certificate verification
)

llm = lm.OpenAIGPT(config)
```

!!! warning "Security Notice"
    Disabling SSL verification makes your connection vulnerable to man-in-the-middle attacks. Only use this in trusted environments.

### 2. HTTP Client Configuration Dictionary

For common scenarios like proxies or custom certificates, use a configuration dictionary:

```python
import langroid.language_models as lm

config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_config={
        "verify": False,  # Or path to CA bundle: "/path/to/ca-bundle.pem"
        "proxy": "http://proxy.company.com:8080",
        "timeout": 30.0,
        "headers": {
            "User-Agent": "MyApp/1.0"
        }
    }
)

llm = lm.OpenAIGPT(config)
```

**Benefits**: This approach enables client caching, improving performance when creating multiple agents.

### 3. Custom HTTP Client Factory

For advanced scenarios requiring dynamic behavior or custom authentication:

```python
import langroid.language_models as lm
from httpx import Client

def create_custom_client():
    """Factory function to create a custom HTTP client."""
    client = Client(
        verify="/path/to/corporate-ca-bundle.pem",
        proxies={
            "http": "http://proxy.corp.com:8080",
            "https": "http://proxy.corp.com:8080"
        },
        timeout=30.0
    )

    # Add custom event hooks for logging
    def log_request(request):
        print(f"API Request: {request.method} {request.url}")

    client.event_hooks = {"request": [log_request]}

    return client

config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_factory=create_custom_client
)

llm = lm.OpenAIGPT(config)
```

If you are using `async` methods, return a tuple of `(Client, AsyncClient)` from your factory:

```python
from httpx import AsyncClient, Client

def create_custom_client():
    """Factory function to create a custom sync and async HTTP clients."""
    client_args = {
        "verify": "/path/to/corporate-ca-bundle.pem",
        "proxy": "http://proxy.corp.com:8080",
        "timeout": 30.0,
    }
    client = Client(**client_args)
    async_client = AsyncClient(**client_args)

    return client, async_client
```

**Note**: Custom factories bypass client caching. Each `OpenAIGPT` instance creates a new client.

## Priority Order

When multiple options are specified, they are applied in this order:
1. `http_client_factory` (highest priority)
2. `http_client_config`
3. `http_verify_ssl` (lowest priority)

## Common Use Cases

### Corporate Proxy with Custom CA Certificate

```python
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_config={
        "verify": "/path/to/corporate-ca-bundle.pem",
        "proxies": {
            "http": "http://proxy.corp.com:8080",
            "https": "https://proxy.corp.com:8443"
        }
    }
)
```

### Debugging API Calls

```python
def debug_client_factory():
    from httpx import Client

    client = Client(verify=False)

    def log_response(response):
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")

    client.event_hooks = {
        "response": [log_response]
    }

    return client

config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_factory=debug_client_factory
)
```

### Local Development with Self-Signed Certificates

```python
# For local OpenAI-compatible APIs
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    api_base="https://localhost:8443/v1",
    http_verify_ssl=False
)
```


## Best Practices

1. **Use the simplest option that meets your needs**:
   - Development/testing: `http_verify_ssl=False`
   - Corporate environments: `http_client_config` with proper CA bundle
   - Complex requirements: `http_client_factory`

2. **Prefer configuration over factories for better performance** - configured clients are cached and reused

3. **Always use proper CA certificates in production** instead of disabling SSL verification

4. **Test your configuration** with a simple API call before deploying:
   ```python
   llm = lm.OpenAIGPT(config)
   response = llm.chat("Hello")
   print(response.content)
   ```

## Troubleshooting

### SSL Certificate Errors
```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
```
**Solution**: Use one of the three configuration options above.


### Proxy Connection Issues
- Verify proxy URL format: `http://proxy:port` or `https://proxy:port`
- Check if proxy requires authentication
- Ensure proxy allows connections to `api.openai.com`

## See Also

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - Official OpenAI documentation