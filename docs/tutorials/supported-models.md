
# Langroid Supported LLMs and Providers

Langroid supports a wide range of Language Model providers through its 
[`OpenAIGPTConfig`][langroid.language_models.openai_gpt.OpenAIGPTConfig] class. 

!!! note "OpenAIGPTConfig is not just for OpenAI models!"
    The `OpenAIGPTConfig` class is a generic configuration class that can be used
    to configure any LLM provider that is OpenAI API-compatible.
    This includes both local and remote models.

You would typically set up the `OpenAIGPTConfig` class with the `chat_model`
parameter, which specifies the model you want to use, and other 
parameters such as `max_output_tokens`, `temperature`, etc
(see the 
[`OpenAIGPTConfig`][langroid.language_models.openai_gpt.OpenAIGPTConfig] class
and its parent class 
[`LLModelConfig`][langroid.language_models.base.LLMConfig] for
full parameter details):



```python
import langroid.language_models as lm
llm_config = lm.OpenAIGPTConfig(
    chat_model="<model-name>", # possibly includes a <provider-name> prefix
    api_key="api-key", # optional, prefer setting in environment variables
    # ... other params such as max_tokens, temperature, etc.
)
```

Below are `chat_model` examples for each supported provider.
For more details see the guides on setting up Langroid with 
[local](https://langroid.github.io/langroid/tutorials/local-llm-setup/) 
and [non-OpenAI LLMs](https://langroid.github.io/langroid/tutorials/non-openai-llms/).
Once you set up the `OpenAIGPTConfig`, you can then directly interact with the LLM,
or set up an Agent with this LLM, and use it by itself, or in a multi-agent setup,
as shown in the [Langroid quick tour](https://langroid.github.io/langroid/tutorials/langroid-tour/)


Although we support specifying the `api_key` directly in the config
(not recommended for security reasons),
more typically you would set the `api_key` in your environment variables.
Below is a table showing for each provider, an example `chat_model` setting, 
and which environment variable to set for the API key.




| Provider      | `chat_model` Example                                     | API Key Environment Variable |
|---------------|----------------------------------------------------------|----------------------------|
| OpenAI        | `gpt-4o`                                                 | `OPENAI_API_KEY` |
| Groq          | `groq/llama3.3-70b-versatile`                            | `GROQ_API_KEY` |
| Cerebras      | `cerebras/llama-3.3-70b`                                 | `CEREBRAS_API_KEY` |
| Gemini        | `gemini/gemini-2.0-flash`                                | `GEMINI_API_KEY` |
| DeepSeek      | `deepseek/deepseek-reasoner`                             | `DEEPSEEK_API_KEY` |
| GLHF          | `glhf/hf:Qwen/Qwen2.5-Coder-32B-Instruct`                | `GLHF_API_KEY` |
| OpenRouter    | `openrouter/deepseek/deepseek-r1-distill-llama-70b:free` | `OPENROUTER_API_KEY` |
| AI/ML API     | `aimlapi/gpt-3.5-turbo`                                   | `AIML_API_KEY` |
| Ollama        | `ollama/qwen2.5`                                         | `OLLAMA_API_KEY` (usually `ollama`) |
| VLLM          | `vllm/mistral-7b-instruct`                               | `VLLM_API_KEY` |
| LlamaCPP      | `llamacpp/localhost:8080`                                | `LLAMA_API_KEY` |
| Generic Local | `local/localhost:8000/v1`                                | No specific env var required |
| LiteLLM       | `litellm/anthropic/claude-3-7-sonnet`                    | Depends on provider |
|               | `litellm/mistral-small`                                  | Depends on provider |
| HF Template   | `local/localhost:8000/v1//mistral-instruct-v0.2`         | Depends on provider |
|               | `litellm/ollama/mistral//hf`                             | |

## HuggingFace Chat Template Formatting

For models requiring specific prompt formatting:

```python
import langroid.language_models as lm

# Specify formatter directly
llm_config = lm.OpenAIGPTConfig(
    chat_model="local/localhost:8000/v1//mistral-instruct-v0.2",
    formatter="mistral-instruct-v0.2"
)

# Using HF formatter auto-detection
llm_config = lm.OpenAIGPTConfig(
    chat_model="litellm/ollama/mistral//hf",
)
```