# Portkey Integration

Langroid provides seamless integration with [Portkey](https://portkey.ai), a powerful AI gateway that enables you to access multiple LLM providers through a unified API with advanced features like caching, retries, fallbacks, and comprehensive observability.

## What is Portkey?

Portkey is an AI gateway that sits between your application and various LLM providers, offering:

- **Unified API**: Access 200+ models from different providers through one interface
- **Reliability**: Automatic retries, fallbacks, and load balancing
- **Observability**: Detailed logging, tracing, and analytics
- **Performance**: Intelligent caching and request optimization
- **Security**: Virtual keys and advanced access controls
- **Cost Management**: Usage tracking and budget controls

For complete documentation, visit the [Portkey Documentation](https://docs.portkey.ai).

## Quick Start

### 1. Setup

First, sign up for a Portkey account at [portkey.ai](https://portkey.ai) and get your API key.

Set up your environment variables, either explicitly or in your `.env` file as usual: 

```bash
# Required: Portkey API key
export PORTKEY_API_KEY="your-portkey-api-key"

# Required: Provider API keys (for the models you want to use)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
# ... other provider keys as needed
```

### 2. Basic Usage

```python
import langroid as lr
import langroid.language_models as lm
from langroid.language_models.provider_params import PortkeyParams

# Create an LLM config to use Portkey's OpenAI-compatible API
# (Note that the name `OpenAIGPTConfig` does NOT imply it only works with OpenAI models;
# the name reflects the fact that the config is meant to be used with an
# OpenAI-compatible API, which Portkey provides for multiple LLM providers.)
llm_config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    portkey_params=PortkeyParams(
        api_key="your-portkey-api-key",  # Or set PORTKEY_API_KEY env var
    )
)

# Create LLM instance
llm = lm.OpenAIGPT(llm_config)

# Use normally
response = llm.chat("What is the smallest prime number?")
print(response.message)
```

### 3. Multiple Providers

Switch between providers seamlessly:

```python
# OpenAI
config_openai = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o",
)

# Anthropic
config_anthropic = lm.OpenAIGPTConfig(
    chat_model="portkey/anthropic/claude-3-5-sonnet-20241022",
)

# Google Gemini
config_gemini = lm.OpenAIGPTConfig(
    chat_model="portkey/google/gemini-2.0-flash-lite",
)
```

## Advanced Features

### Virtual Keys

Use virtual keys to abstract provider management:

```python
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o",
    portkey_params=PortkeyParams(
        virtual_key="vk-your-virtual-key",  # Configured in Portkey dashboard
    )
)
```

### Caching and Performance

Enable intelligent caching to reduce costs and improve performance:

```python
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    portkey_params=PortkeyParams(
        cache={
            "enabled": True,
            "ttl": 3600,  # 1 hour cache
            "namespace": "my-app"
        },
        cache_force_refresh=False,
    )
)
```

### Retry Strategies

Configure automatic retries for better reliability:

```python
config = lm.OpenAIGPTConfig(
    chat_model="portkey/anthropic/claude-3-haiku-20240307",
    portkey_params=PortkeyParams(
        retry={
            "max_retries": 3,
            "backoff": "exponential",
            "jitter": True
        }
    )
)
```

### Observability and Tracing

Add comprehensive tracking for production monitoring:

```python
import uuid

config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o",
    portkey_params=PortkeyParams(
        trace_id=f"trace-{uuid.uuid4().hex[:8]}",
        metadata={
            "user_id": "user-123",
            "session_id": "session-456",
            "app_version": "1.2.3"
        },
        user="user-123",
        organization="my-org",
        custom_headers={
            "x-request-source": "langroid",
            "x-feature": "chat-completion"
        }
    )
)
```

## Configuration Reference

The `PortkeyParams` class supports all Portkey features:

```python
from langroid.language_models.provider_params import PortkeyParams

params = PortkeyParams(
    # Authentication
    api_key="pk-...",                    # Portkey API key
    virtual_key="vk-...",               # Virtual key (optional)
    
    # Observability
    trace_id="trace-123",               # Request tracing
    metadata={"key": "value"},          # Custom metadata
    user="user-id",                     # User identifier
    organization="org-id",              # Organization identifier
    
    # Performance
    cache={                             # Caching configuration
        "enabled": True,
        "ttl": 3600,
        "namespace": "my-app"
    },
    cache_force_refresh=False,          # Force cache refresh
    
    # Reliability
    retry={                             # Retry configuration
        "max_retries": 3,
        "backoff": "exponential",
        "jitter": True
    },
    
    # Custom headers
    custom_headers={                    # Additional headers
        "x-custom": "value"
    },
    
    # Base URL (usually not needed)
    base_url="https://api.portkey.ai"   # Portkey API endpoint
)
```

## Supported Providers

Portkey supports 200+ models from various providers. Common ones include:

```python
# OpenAI
"portkey/openai/gpt-4o"
"portkey/openai/gpt-4o-mini"

# Anthropic
"portkey/anthropic/claude-3-5-sonnet-20241022"
"portkey/anthropic/claude-3-haiku-20240307"

# Google
"portkey/google/gemini-2.0-flash-lite"
"portkey/google/gemini-1.5-pro"

# Cohere
"portkey/cohere/command-r-plus"

# Meta
"portkey/meta/llama-3.1-405b-instruct"

# And many more...
```

Check the [Portkey documentation](https://docs.portkey.ai/docs/integrations/models) for the complete list.

## Examples

Langroid includes comprehensive Portkey examples in `examples/portkey/`:

1. **`portkey_basic_chat.py`** - Basic usage with multiple providers
2. **`portkey_advanced_features.py`** - Caching, retries, and observability
3. **`portkey_multi_provider.py`** - Comparing responses across providers

Run any example:

```bash
cd examples/portkey
python portkey_basic_chat.py
```

## Best Practices

### 1. Use Environment Variables

Never hardcode API keys:

```bash
# .env file
PORTKEY_API_KEY=your_portkey_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### 2. Implement Fallback Strategies

Use multiple providers for reliability:

```python
providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-haiku-20240307"),
    ("google", "gemini-2.0-flash-lite")
]

for provider, model in providers:
    try:
        config = lm.OpenAIGPTConfig(
            chat_model=f"portkey/{provider}/{model}"
        )
        llm = lm.OpenAIGPT(config)
        return llm.chat(question)
    except Exception:
        continue  # Try next provider
```

### 3. Add Meaningful Metadata

Include context for better observability:

```python
params = PortkeyParams(
    metadata={
        "user_id": user.id,
        "feature": "document_qa",
        "document_type": "pdf",
        "processing_stage": "summary"
    }
)
```

### 4. Use Caching Wisely

Enable caching for deterministic queries:

```python
# Good for caching
params = PortkeyParams(
    cache={"enabled": True, "ttl": 3600}
)

# Use with deterministic prompts
response = llm.chat("What is the capital of France?")
```

### 5. Monitor Performance

Use trace IDs to track request flows:

```python
import uuid

trace_id = f"trace-{uuid.uuid4().hex[:8]}"
params = PortkeyParams(
    trace_id=trace_id,
    metadata={"operation": "document_processing"}
)

# Use the same trace_id for related requests
```

## Monitoring and Analytics

### Portkey Dashboard

View detailed analytics at [app.portkey.ai](https://app.portkey.ai):

- Request/response logs
- Token usage and costs
- Performance metrics (latency, errors)
- Provider comparisons
- Custom filters by metadata

### Custom Filtering

Use metadata and headers to filter requests:

```python
# Tag requests by feature
params = PortkeyParams(
    metadata={"feature": "chat", "version": "v2"},
    custom_headers={"x-request-type": "production"}
)
```

Then filter in the dashboard by:
- `metadata.feature = "chat"`
- `headers.x-request-type = "production"`

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   Error: Unauthorized (401)
   ```
   - Check `PORTKEY_API_KEY` is set correctly
   - Verify API key is active in Portkey dashboard

2. **Provider API Key Missing**
   ```
   Error: Missing API key for provider
   ```
   - Set provider API key (e.g., `OPENAI_API_KEY`)
   - Or use virtual keys in Portkey dashboard

3. **Model Not Found**
   ```
   Error: Model not supported
   ```
   - Check model name format: `portkey/provider/model`
   - Verify model is available through Portkey

4. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   - Configure retry parameters
   - Use virtual keys for better rate limit management

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("langroid").setLevel(logging.DEBUG)
```

### Test Configuration

Verify your setup:

```python
# Test basic connection
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    max_output_tokens=50
)
llm = lm.OpenAIGPT(config)
response = llm.chat("Hello")
print("✅ Portkey integration working!")
```

## Migration Guide

### From Direct Provider Access

If you're currently using providers directly:

```python
# Before: Direct OpenAI
config = lm.OpenAIGPTConfig(
    chat_model="gpt-4o-mini"
)

# After: Through Portkey
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini"
)
```

### Adding Advanced Features Gradually

Start simple and add features as needed:

```python
# Step 1: Basic Portkey
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini"
)

# Step 2: Add caching
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    portkey_params=PortkeyParams(
        cache={"enabled": True, "ttl": 3600}
    )
)

# Step 3: Add observability
config = lm.OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    portkey_params=PortkeyParams(
        cache={"enabled": True, "ttl": 3600},
        metadata={"app": "my-app", "user": "user-123"},
        trace_id="trace-abc123"
    )
)
```

## Resources

- **Portkey Website**: [https://portkey.ai](https://portkey.ai)
- **Portkey Documentation**: [https://docs.portkey.ai](https://docs.portkey.ai)
- **Portkey Dashboard**: [https://app.portkey.ai](https://app.portkey.ai)
- **Supported Models**: [https://docs.portkey.ai/docs/integrations/models](https://docs.portkey.ai/docs/integrations/models)
- **Langroid Examples**: `examples/portkey/` directory
- **API Reference**: [https://docs.portkey.ai/docs/api-reference](https://docs.portkey.ai/docs/api-reference)