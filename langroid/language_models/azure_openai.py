import os

import openai
from dotenv import load_dotenv

from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPT,
    OpenAIGPTConfig,
)


class AzureConfig(OpenAIGPTConfig):
    """
    Configuration for Azure OpenAI GPT.

    Attributes:
        type (str): should be ``azure.``
        api_version (str): can be set in the ``.env`` file as
            ``AZURE_OPENAI_API_VERSION.``
        deployment_name (str): can be set in the ``.env`` file as
            ``AZURE_OPENAI_DEPLOYMENT_NAME`` and should be based the custom name you
            chose for your deployment when you deployed a model.
        model_name (str): can be set in the ``.env`` file as ``AZURE_GPT_MODEL_NAME``
            and should be based on the model name chosen during setup.
    """

    type: str = "azure"
    api_version: str = "2023-05-15"
    deployment_name: str = ""
    model_name: str = ""


class AzureGPT(OpenAIGPT):
    """
    Class to access OpenAI LLMs via Azure. These env variables can be obtained from the
    file `.azure_env`. Azure OpenAI doesn't support ``completion``
    Attributes:
        config (AzureConfig): AzureConfig object
        api_key (str): Azure API key
        api_base (str): Azure API base url
        api_version (str): Azure API version
        model_name (str): the name of gpt model in your deployment
    """

    def __init__(self, config: AzureConfig):
        super().__init__(config)
        self.config: AzureConfig = config
        self.api_type = config.type
        openai.api_type = self.api_type
        load_dotenv()
        self.api_key = os.getenv("AZURE_API_KEY", "")
        if self.api_key == "":
            raise ValueError(
                """
                AZURE_API_KEY not set in .env file,
                please set it to your Azure API key."""
            )

        self.api_base = os.getenv("AZURE_OPENAI_API_BASE", "")
        if self.api_base == "":
            raise ValueError(
                """
                AZURE_OPENAI_API_BASE not set in .env file,
                please set it to your Azure API key."""
            )
        # we don't need this for ``api_key`` because it's handled inside
        # ``openai_gpt.py`` methods before invoking chat/completion calls
        else:
            openai.api_base = self.api_base

        self.api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION", "") or config.api_version
        )
        openai.api_version = self.api_version

        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
        if self.deployment_name == "":
            raise ValueError(
                """
                AZURE_OPENAI_DEPLOYMENT_NAME not set in .env file,
                please set it to your Azure openai deployment name."""
            )

        self.model_name = os.getenv("AZURE_GPT_MODEL_NAME", "")
        if self.model_name == "":
            raise ValueError(
                """
                AZURE_GPT_MODEL_NAME not set in .env file,
                please set it to chat model name in you deployment model."""
            )

        # set the chat model to be the same as the model_name
        # This corresponds to the gpt model you chose for your deployment
        # when you deployed a model
        if "35-turbo" in self.model_name:
            self.config.chat_model = OpenAIChatModel.GPT3_5_TURBO
        else:
            self.config.chat_model = OpenAIChatModel.GPT4
