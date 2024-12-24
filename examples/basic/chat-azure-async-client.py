"""
Example showing how to use Langroid with Azure OpenAI and Entra ID
authentication by providing a custom client.

This is an async version of the example in chat-azure-client.py.

For more details see here:
https://langroid.github.io/langroid/notes/custom-azure-client/
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity

"""

import os

import azure.identity as azure_identity
import azure.identity.aio as azure_identity_async
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

import langroid as lr
import langroid.language_models as lm

load_dotenv()


def get_azure_openai_client():
    return AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        azure_ad_token_provider=azure_identity.get_bearer_token_provider(
            azure_identity.DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        ),
    )


def get_azure_openai_async_client():
    return AsyncAzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        azure_ad_token_provider=azure_identity_async.get_bearer_token_provider(
            azure_identity_async.DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        ),
    )


lm_config = lm.AzureConfig(
    azure_openai_client_provider=get_azure_openai_client,
    azure_openai_async_client_provider=get_azure_openai_async_client,
)


async def main():
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=lm_config))
    task = lr.Task(agent, interactive=False)
    response = await task.run_async(
        "Who is the president of the United States? Reply and end with DONE"
    )
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
