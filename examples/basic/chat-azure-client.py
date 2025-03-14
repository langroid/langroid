"""
Example showing how to use Langroid with Azure OpenAI and Entra ID
authentication by providing a custom client.

NOTE: this example is ONLY meant for those who are trying to use a custom
Azure client, as in this scenario:
https://langroid.github.io/langroid/notes/custom-azure-client/
This NOT TYPICAL for most users, and should be ignored if you are not using such a
custom client.

For typical usage of Azure-deployed models with Langroid, see
the [`test_azure_openai.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_azure_openai.py) and
[`example/basic/chat.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat.py)


For an async version of this example, see chat-azure-async-client.py.

For more details see here:
https://langroid.github.io/langroid/notes/custom-azure-client/
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity

"""

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

import langroid as lr
import langroid.language_models as lm

load_dotenv()


def get_azure_openai_client():
    return AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        ),
    )


lm_config = lm.AzureConfig(
    azure_openai_client_provider=get_azure_openai_client,
)

if __name__ == "__main__":
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=lm_config))
    task = lr.Task(agent, interactive=False)
    task.run("Who is the president of the United States? Reply and end with DONE")
