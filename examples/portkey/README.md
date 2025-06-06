# Portkey Examples

This folder contains examples demonstrating how to use [Portkey](https://portkey.ai) with Langroid for enhanced LLM gateway functionality and observability.

## Prerequisites

Before running any examples, make sure you've installed Langroid as usual.

At minimum, have these environment variables set up in your `.env` file or environment:
```bash
PORTKEY_API_KEY=your_portkey_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # or any provider's key
ANTHROPIC_API_KEY=your_anthropic_key_here  # if using Anthropic
```

### 1. Portkey Basic Chat (`portkey_basic_chat.py`)

Demonstrates basic chat functionality with Portkey:
- Uses Portkey as a gateway to different AI providers
- Shows automatic provider API key resolution
- Demonstrates model switching across providers

```python
# Run the example from root of repo, after activating your virtual environment with uv:
uv run examples/portkey/portkey_basic_chat.py
```

### 2. Portkey Advanced Features (`portkey_advanced_features.py`)

Shows how to use Portkey's advanced features:
- Virtual keys for provider management
- Caching and retry configurations
- Request tracing and metadata
- Custom headers for observability

```python
# Run the example
uv run examples/portkey/portkey_advanced_features.py
```

### 3. Portkey Multi-Provider Example (`portkey_multi_provider.py`)

Showcases Portkey's ability to switch between providers:
- Compares responses from different providers
- Demonstrates fallback strategies
- Shows how to use virtual keys for different models

```python
# Run the example
uv run examples/portkey/portkey_multi_provider.py
```

## Using Portkey

### Basic Configuration

Portkey can route requests to any AI provider through a unified API:

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.language_models.provider_params import PortkeyParams

# Configure for OpenAI via Portkey
config = OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o-mini",
    portkey_params=PortkeyParams(
        api_key="your-portkey-api-key",  # Or use PORTKEY_API_KEY env var
    )
)

# Configure for Anthropic via Portkey
config = OpenAIGPTConfig(
    chat_model="portkey/anthropic/claude-3-sonnet-20240229",
    portkey_params=PortkeyParams(
        api_key="your-portkey-api-key",
    )
)
```

### Advanced Features

Portkey provides powerful gateway features:

```python
from langroid.language_models.provider_params import PortkeyParams

# Advanced configuration with observability
params = PortkeyParams(
    api_key="your-portkey-api-key",
    virtual_key="vk-your-virtual-key",  # For provider abstraction
    trace_id="trace-123",               # For request tracing
    metadata={"user": "john", "app": "langroid"},  # Custom metadata
    retry={"max_retries": 3, "backoff": "exponential"},  # Retry config
    cache={"enabled": True, "ttl": 3600},  # Caching config
    cache_force_refresh=False,          # Cache control
    user="user-123",                    # User identifier
    organization="org-456",             # Organization identifier
    custom_headers={                    # Additional custom headers
        "x-custom-header": "value"
    }
)

config = OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o",
    portkey_params=params
)
```

### Supported Providers

Portkey supports many AI providers:

```python
# OpenAI
chat_model="portkey/openai/gpt-4o"

# Anthropic
chat_model="portkey/anthropic/claude-3-5-sonnet-20241022"

# Google Gemini
chat_model="portkey/google/gemini-2.0-flash-lite"

# Cohere
chat_model="portkey/cohere/command-r-plus"

# Many more providers available through Portkey
```

### Environment Variables

Portkey integration automatically resolves API keys from environment variables:

```bash
# Portkey API key
PORTKEY_API_KEY=your_portkey_api_key

# Provider API keys (used for actual model calls)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key
```

### Virtual Keys

Use virtual keys to abstract provider management:

```python
# Configure with virtual key
config = OpenAIGPTConfig(
    chat_model="portkey/openai/gpt-4o",
    portkey_params=PortkeyParams(
        virtual_key="vk-your-virtual-key",  # Manages provider key automatically
    )
)
```

### Viewing Results

1. Visit the [Portkey Dashboard](https://app.portkey.ai)
2. Navigate to your project
3. View detailed analytics:
   - Request/response logs
   - Token usage and costs
   - Performance metrics
   - Error rates and debugging

## Best Practices

1. **Use Virtual Keys**: Abstract provider management for easier switching
2. **Add Metadata**: Include user and application context for better tracking
3. **Configure Retries**: Set up automatic retry strategies for reliability
4. **Enable Caching**: Reduce costs and improve performance with intelligent caching
5. **Monitor Performance**: Use trace IDs and metadata for detailed observability

## Troubleshooting

Common issues and solutions:

1. **Authentication Errors**:
   - Verify `PORTKEY_API_KEY` is set correctly
   - Ensure provider API keys are available (e.g., `OPENAI_API_KEY`)

2. **Model Not Found**:
   - Ensure the model name includes the `portkey/` prefix
   - Verify the provider and model are supported by Portkey

3. **Rate Limiting**:
   - Configure retry parameters in PortkeyParams
   - Use virtual keys for better rate limit management

4. **Virtual Key Issues**:
   - Verify virtual key is correctly configured in Portkey dashboard
   - Check virtual key has access to the requested provider/model

For more help, visit the [Portkey Documentation](https://docs.portkey.ai).