# Custom Azure OpenAI client

!!! warning "This is only for using a Custom Azure OpenAI client"
    This note **only** meant for those who are trying to use a custom Azure client,
    and is NOT TYPICAL for most users. For typical usage of Azure-deployed models with Langroid, see
    the [docs](https://langroid.github.io/langroid/notes/azure-openai-models/), 
    the [`test_azure_openai.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_azure_openai.py) and
    [`example/basic/chat.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat.py)


Example showing how to use Langroid with Azure OpenAI and Entra ID
authentication by providing a custom client.

By default, Langroid manages the configuration and creation 
of the Azure OpenAI client (see the [Setup guide](https://langroid.github.io/langroid/quick-start/setup/#microsoft-azure-openai-setupoptional)
for details). In most cases, the available configuration options
are sufficient, but if you need to manage any options that
are not exposed, you instead have the option of providing a custom
client, in Langroid v0.29.0 and later. 

In order to use a custom client, you must provide a function that
returns the configured client. Depending on whether you need to make
synchronous or asynchronous calls, you need to provide the appropriate
client. A sketch of how this is done (supporting both sync and async calls)
is given below:

```python
def get_azure_openai_client():
    return AzureOpenAI(...)

def get_azure_openai_async_client():
    return AsyncAzureOpenAI(...)

lm_config = lm.AzureConfig(
    azure_openai_client_provider=get_azure_openai_client,
    azure_openai_async_client_provider=get_azure_openai_async_client,
)
```

## Microsoft Entra ID Authentication

A key use case for a custom client is [Microsoft Entra ID authentication](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity).
Here you need to provide an `azure_ad_token_provider` to the client. 
For examples on this, see [examples/basic/chat-azure-client.py](https://github.com/langroid/langroid/blob/main/examples/basic/chat-azure-client.py) 
and [examples/basic/chat-azure-async-client.py](https://github.com/langroid/langroid/blob/main/examples/basic/chat-azure-async-client.py).
