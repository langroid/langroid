from dotenv import load_dotenv
from httpx import Timeout
from openai import AsyncAzureOpenAI, AzureOpenAI

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
        model_version (str): can be set in the ``.env`` file as
          ``AZURE_OPENAI_MODEL_VERSION`` and should be based on the model name
          chosen during setup.
    """

    api_key: str = ""  # CAUTION: set this ONLY via env var AZURE_OPENAI_API_KEY
    type: str = "azure"
    api_version: str = "2023-05-15"
    deployment_name: str = ""
    model_name: str = ""
    model_version: str = ""  # is used to determine the cost of using the model
    api_base: str = ""

    # all of the vars above can be set via env vars,
    # by upper-casing the name and prefixing with `env_prefix`, e.g.
    # AZURE_OPENAI_API_VERSION=2023-05-15
    # This is either done in the .env file, or via an explicit
    # `export AZURE_OPENAI_API_VERSION=...`
    class Config:
        env_prefix = "AZURE_OPENAI_"


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
        model_version (str): the version of gpt model in your deployment
    """

    def __init__(self, config: AzureConfig):
        # This will auto-populate config values from .env file
        load_dotenv()
        super().__init__(config)
        self.config: AzureConfig = config
        if self.config.api_key == "":
            raise ValueError(
                """
                AZURE_OPENAI_API_KEY not set in .env file,
                please set it to your Azure API key."""
            )

        if self.config.api_base == "":
            raise ValueError(
                """
                AZURE_OPENAI_API_BASE not set in .env file,
                please set it to your Azure API key."""
            )

        if self.config.deployment_name == "":
            raise ValueError(
                """
                AZURE_OPENAI_DEPLOYMENT_NAME not set in .env file,
                please set it to your Azure openai deployment name."""
            )
        self.deployment_name = self.config.deployment_name

        if self.config.model_name == "":
            raise ValueError(
                """
                AZURE_OPENAI_MODEL_NAME not set in .env file,
                please set it to chat model name in your deployment."""
            )

        # set the chat model to be the same as the model_name
        # This corresponds to the gpt model you chose for your deployment
        # when you deployed a model
        self.set_chat_model()

        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.api_base,
            api_version=self.config.api_version,
            azure_deployment=self.config.deployment_name,
        )
        self.async_client = AsyncAzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.api_base,
            api_version=self.config.api_version,
            azure_deployment=self.config.deployment_name,
            timeout=Timeout(self.config.timeout),
        )

    def set_chat_model(self) -> None:
        """
        Sets the chat model configuration based on the model name specified in the
        ``.env``. This function checks the `model_name` in the configuration and sets
        the appropriate chat model in the `config.chat_model`. It supports handling for
        '35-turbo' and 'gpt-4' models. For 'gpt-4', it further delegates the handling
        to `handle_gpt4_model` method. If the model name does not match any predefined
        models, it defaults to `OpenAIChatModel.GPT4`.
        """
        MODEL_35_TURBO = "35-turbo"
        MODEL_GPT4 = "gpt-4"

        if self.config.model_name == MODEL_35_TURBO:
            self.config.chat_model = OpenAIChatModel.GPT3_5_TURBO
        elif self.config.model_name == MODEL_GPT4:
            self.handle_gpt4_model()
        else:
            self.config.chat_model = OpenAIChatModel.GPT4

    def handle_gpt4_model(self) -> None:
        """
        Handles the setting of the GPT-4 model in the configuration.
        This function checks the `model_version` in the configuration.
        If the version is not set, it raises a ValueError indicating
        that the model version needs to be specified in the ``.env``
        file.  It sets `OpenAIChatMode.GPT4o` if the version is
        '2024-05-13', `OpenAIChatModel.GPT4_TURBO` if the version is
        '1106-Preview', otherwise, it defaults to setting
        `OpenAIChatModel.GPT4`.
        """
        VERSION_1106_PREVIEW = "1106-Preview"
        VERSION_GPT4o = "2024-05-13"

        if self.config.model_version == "":
            raise ValueError(
                "AZURE_OPENAI_MODEL_VERSION not set in .env file. "
                "Please set it to the chat model version used in your deployment."
            )

        if self.config.model_version == VERSION_GPT4o:
            self.config.chat_model = OpenAIChatModel.GPT4o
        elif self.config.model_version == VERSION_1106_PREVIEW:
            self.config.chat_model = OpenAIChatModel.GPT4_TURBO
        else:
            self.config.chat_model = OpenAIChatModel.GPT4
