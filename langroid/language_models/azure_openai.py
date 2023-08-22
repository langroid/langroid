import os

import openai
from dotenv import load_dotenv

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


class AzureConfig(OpenAIGPTConfig):
    type: str = "azure"
    api_version: str = "2023-07-01-preview"


class AzureGPT(OpenAIGPT):
    """
    Class to access OpenAI LLMs via Azure
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
