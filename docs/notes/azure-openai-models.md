# Azure OpenAI Models

To use OpenAI models deployed on Azure, first ensure a few environment variables
are defined (either in your `.env` file or in your environment):

- `AZURE_OPENAI_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your_resource.openai.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is an OPTIONAL deployment name which may be
  defined by the user during the model setup.
- `AZURE_OPENAI_CHAT_MODEL` Azure OpenAI allows specific model names when you select the model for your deployment. You need to put precisely the exact model name that was selected. For example, GPT-3.5 (should be `gpt-35-turbo-16k` or `gpt-35-turbo`) or GPT-4 (should be `gpt-4-32k` or `gpt-4`).
- `AZURE_OPENAI_MODEL_NAME` (Deprecated, use `AZURE_OPENAI_CHAT_MODEL` instead).

This page [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#environment-variables) 
provides more information on how to obtain these values.

To use an Azure-deployed model in Langroid, you can use the `AzureConfig` class:

```python
import langroid.language_models as lm
import langroid as lr

llm_config = lm.AzureConfig(
    chat_model="gpt-4o"
    # the other settings can be provided explicitly here, 
    # or are obtained from the environment
)
llm = lm.AzureGPT(config=llm_config)

response = llm.chat(
  messages=[
    lm.LLMMessage(role=lm.Role.SYSTEM, content="You are a helpful assistant."),
    lm.LLMMessage(role=lm.Role.USER, content="3+4=?"),
  ]
)

agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        llm=llm_config,
        system_message="You are a helpful assistant.",
    )
)

response = agent.llm_response("is 4 odd?")
print(response.content)  # "Yes, 4 is an even number."
response = agent.llm_response("what about 2?")  # follow-up question
```

## Using Azure OpenAI API v1 with Standard OpenAI Clients

Azure's October 2025 API update allows using standard OpenAI clients instead of
Azure-specific ones. However, Azure deployment names often differ from actual
model identifiers, which can cause issues with model capability detection.

If your deployment name differs from the actual model name, use `chat_model_orig`
to specify the actual model for proper capability detection:

```python
import langroid.language_models as lm

llm_config = lm.OpenAIGPTConfig(
    chat_model="my-gpt4o-deployment",     # Your Azure deployment name
    chat_model_orig="gpt-4o",             # Actual model name for capability detection
    api_base="https://your-resource.openai.azure.com/",
)
```

This ensures Langroid correctly identifies model capabilities (context length,
supported features, etc.) even when the deployment name doesn't match the
underlying model.
