import os

import openai
from dotenv import load_dotenv

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


class AzureConfig(OpenAIGPTConfig):
    """
    Configuration for Azure OpenAI GPT. You need to supply the env vars listed in
    ``.azure_env_template`` after renaming the file to ``.azure_env``. Because this file
    is used by this class to find the env vars.
    Attributes:
        type (str): should be ``azure``
        api_version (str): can be set inside the ``.azure_env``
        deployment_name (str): can be set inside the ``.azure_env`` and should be based
        the custom name you chose for your deployment when you deployed a model
    """

    type: str = "azure"
    api_version: str = "2023-07-01-preview"
    deployment_name: str = ""


class AzureGPT(OpenAIGPT):
    """
    Class to access OpenAI LLMs via Azure. These env variables can be obtained from the
    file `.azure_env`. Azure OpenAI doesn't support ``completion``
    Attributes:
        config: AzureConfig object
        api_key: Azure API key
        api_base: Azure API base url
        api_version: Azure API version
    """

    def __init__(self, config: AzureConfig):
        super().__init__(config)
        self.config: AzureConfig = config
        self.api_type = config.type
        openai.api_type = self.api_type
        load_dotenv(dotenv_path=".azure_env")
        self.api_key = os.getenv("AZURE_API_KEY", "")
        if self.api_key == "":
            raise ValueError(
                """
                AZURE_API_KEY not set in .env file,
                please set it to your Azure API key."""
            )

        self.api_base = os.getenv("OPENAI_API_BASE", "")
        if self.api_base == "":
            raise ValueError(
                """
                OPENAI_API_BASE not set in .env file,
                please set it to your Azure API key."""
            )
        # we don't need this for ``api_key`` because it's handled inside
        # ``openai_gpt.py`` methods before invoking chat/completion calls
        else:
            openai.api_base = self.api_base

        self.api_version = os.getenv("OPENAI_API_VERSION", "")
        if self.api_version == "":
            self.api_version = config.api_version
        openai.api_version = self.api_version

        self.deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME", "")
        if self.deployment_name == "":
            raise ValueError(
                """
                OPENAI_DEPLOYMENT_NAME not set in .env file,
                please set it to your Azure API key."""
            )
