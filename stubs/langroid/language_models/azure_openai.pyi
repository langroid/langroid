from _typeshed import Incomplete

from langroid.language_models.openai_gpt import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPT as OpenAIGPT,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)

class AzureConfig(OpenAIGPTConfig):
    api_key: str
    type: str
    api_version: str
    deployment_name: str
    model_name: str
    model_version: str
    api_base: str

    class Config:
        env_prefix: str

class AzureGPT(OpenAIGPT):
    config: Incomplete
    deployment_name: Incomplete
    client: Incomplete
    async_client: Incomplete
    def __init__(self, config: AzureConfig) -> None: ...
    def set_chat_model(self) -> None: ...
    def handle_gpt4_model(self) -> None: ...
