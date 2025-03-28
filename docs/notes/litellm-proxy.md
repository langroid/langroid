# Using LiteLLM Proxy with OpenAIGPTConfig

You can easily configure Langroid to use LiteLLM proxy for accessing models with a 
simple prefix `litellm-proxy/` in the `chat_model` name:

## Using the `litellm-proxy/` prefix

When you specify a model with the `litellm-proxy/` prefix, Langroid automatically uses the LiteLLM proxy configuration:

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig

config = OpenAIGPTConfig(
    chat_model="litellm-proxy/your-model-name"
)
```

## Setting LiteLLM Proxy Parameters

When using the `litellm-proxy/` prefix, Langroid will read connection parameters from either:

1. The `litellm_proxy` config object:
   ```python
   from langroid.language_models.openai_gpt import OpenAIGPTConfig, LiteLLMProxyConfig
   
   config = OpenAIGPTConfig(
       chat_model="litellm-proxy/your-model-name",
       litellm_proxy=LiteLLMProxyConfig(
           api_key="your-litellm-proxy-api-key",
           api_base="http://your-litellm-proxy-url"
       )
   )
   ```

2. Environment variables (which take precedence):
   ```bash
   export LITELLM_API_KEY="your-litellm-proxy-api-key"
   export LITELLM_API_BASE="http://your-litellm-proxy-url"
   ```

This approach makes it simple to switch between using LiteLLM proxy and 
other model providers by just changing the model name prefix,
without needing to modify the rest of your code or tweaking env variables.

## Note: LiteLLM Proxy vs LiteLLM Library

**Important distinction:** Using the `litellm-proxy/` prefix connects to a LiteLLM proxy server, which is different from using the `litellm/` prefix. The latter utilizes the LiteLLM adapter library directly without requiring a proxy server. Both approaches are supported in Langroid, but they serve different use cases:

- Use `litellm-proxy/` when connecting to a deployed LiteLLM proxy server
- Use `litellm/` when you want to use the LiteLLM library's routing capabilities locally

Choose the approach that best fits your infrastructure and requirements.