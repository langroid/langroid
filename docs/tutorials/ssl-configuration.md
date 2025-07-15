# SSL and HTTP Client Configuration

Langroid provides options to customize the HTTP client used for OpenAI API calls, which is particularly useful in corporate environments with SSL certificate issues or proxy requirements.

## SSL Certificate Verification Issues

If you encounter SSL certificate verification errors when using OpenAI models (common in corporate networks with self-signed certificates), you have three options:

### Option 1: Disable SSL Verification (Quick but Less Secure)

!!! warning "Security Warning"
    Disabling SSL verification makes your connection vulnerable to man-in-the-middle attacks. 
    Only use this in trusted environments like corporate networks with known self-signed certificates.

```python
import langroid.language_models as lm

# Disable SSL verification
llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_verify_ssl=False  # Disables SSL certificate verification
)

# Use with an agent
agent = lr.ChatAgent(lr.ChatAgentConfig(llm=llm_config))
```


### Option 2: HTTP Client Configuration (Moderate Control, Cacheable)

For common scenarios like proxies and SSL settings, you can use `http_client_config`. This approach allows client caching for better performance:

```python
import langroid.language_models as lm

# Configure HTTP client with a dictionary
llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_config={
        "verify": False,  # or path to CA bundle: "/path/to/ca-bundle.pem"
        "proxy": "http://proxy.company.com:8080",  # or use "proxies" dict
        "timeout": 30.0,
        "headers": {
            "User-Agent": "MyApp/1.0"
        }
    }
)

# This configuration is cacheable - multiple agents can share the same client
agent1 = lr.ChatAgent(lr.ChatAgentConfig(llm=llm_config))
agent2 = lr.ChatAgent(lr.ChatAgentConfig(llm=llm_config))  # Reuses cached client
```

### Option 3: Custom HTTP Client Factory (Maximum Control)

For the most complex scenarios requiring dynamic behavior, custom authentication, or event hooks, use a factory function:

```python
import langroid.language_models as lm
from httpx import Client, AsyncClient

def create_custom_http_client():
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
    client.event_hooks = {
        "request": [log_request],
        "response": [log_response]
    }
    
    return client

# Use the custom client factory (not cacheable)
llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_factory=create_custom_http_client
)
```

!!! note "Client Caching"
    - Options 1 & 2 (`http_verify_ssl` and `http_client_config`) support client caching
    - Option 3 (`http_client_factory`) does NOT cache clients - each instance creates a new client
    - Choose based on your performance vs flexibility needs

## Common Use Cases

### Corporate Proxy with Self-Signed Certificate

```python
import langroid.language_models as lm
from httpx import Client
import ssl

def create_corporate_client():
    # Create SSL context with custom CA bundle
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations("/path/to/corporate-ca-bundle.pem")
    
    return Client(
        verify=ssl_context,
        proxies={
            "http": "http://proxy.corp.com:8080",
            "https": "http://proxy.corp.com:8080"
        },
        headers={
            "Proxy-Authorization": "Basic <encoded-credentials>"
        }
    )

llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    http_client_factory=create_corporate_client
)
```

### Development/Testing with Local API Server

```python
import langroid.language_models as lm

# For local development with self-signed certificates
llm_config = lm.OpenAIGPTConfig(
    chat_model="gpt-4",
    api_base="https://localhost:8443/v1",
    http_verify_ssl=False  # OK for local development
)
```

## Important Notes

1. **Client Caching**: 
   - `http_verify_ssl` and `http_client_config` create cacheable clients (better performance)
   - `http_client_factory` does NOT cache clients (more flexibility but higher overhead)

2. **Priority Order**: When multiple options are specified:
   - `http_client_factory` (highest priority)
   - `http_client_config` 
   - `http_verify_ssl` (lowest priority)

3. **Async Support**: If you need async operations with a custom client, ensure your factory returns
   an appropriate async client (httpx.AsyncClient).

4. **Security Best Practices**:
   - Never disable SSL verification in production unless absolutely necessary
   - If you must disable SSL, ensure you're only connecting to known, trusted endpoints
   - Consider using a custom CA bundle instead of disabling verification entirely
   - Regularly review and update your SSL configuration

## Troubleshooting

### SSL: CERTIFICATE_VERIFY_FAILED

If you see this error:
```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Try these solutions in order:
1. Update your system's certificate store
2. Use a custom CA bundle with your corporate certificates
3. As a last resort, use `http_verify_ssl=False`


### Proxy Connection Issues

If you're behind a corporate proxy and getting connection errors:
1. Ensure your proxy settings are correct
2. Check if you need proxy authentication
3. Verify the proxy allows HTTPS connections to api.openai.com