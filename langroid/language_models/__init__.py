from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import utils
    from . import config
    from . import base
    from . import openai_gpt
    from . import azure_openai
    from . import prompt_formatter

    from .base import (
        LLMConfig,
        LLMMessage,
        LLMFunctionCall,
        LLMFunctionSpec,
        Role,
        LLMTokenUsage,
        LLMResponse,
    )
    from .openai_gpt import (
        OpenAIChatModel,
        OpenAICompletionModel,
        OpenAIGPTConfig,
        OpenAIGPT,
    )
    from .azure_openai import AzureConfig, AzureGPT

else:
    utils = LazyLoad("langroid.language_models.utils")
    config = LazyLoad("langroid.language_models.config")
    base = LazyLoad("langroid.language_models.base")
    openai_gpt = LazyLoad("langroid.language_models.openai_gpt")
    azure_openai = LazyLoad("langroid.language_models.azure_openai")
    prompt_formatter = LazyLoad("langroid.language_models.prompt_formatter")

    LLMConfig = LazyLoad("langroid.language_models.base.LLMConfig")
    LLMMessage = LazyLoad("langroid.language_models.base.LLMMessage")
    LLMFunctionCall = LazyLoad("langroid.language_models.base.LLMFunctionCall")
    LLMFunctionSpec = LazyLoad("langroid.language_models.base.LLMFunctionSpec")
    Role = LazyLoad("langroid.language_models.base.Role")
    LLMTokenUsage = LazyLoad("langroid.language_models.base.LLMTokenUsage")
    LLMResponse = LazyLoad("langroid.language_models.base.LLMResponse")

    OpenAIChatModel = LazyLoad("langroid.language_models.openai_gpt.OpenAIChatModel")
    OpenAICompletionModel = LazyLoad(
        "langroid.language_models.openai_gpt.OpenAICompletionModel"
    )
    OpenAIGPTConfig = LazyLoad("langroid.language_models.openai_gpt.OpenAIGPTConfig")
    OpenAIGPT = LazyLoad("langroid.language_models.openai_gpt.OpenAIGPT")

    AzureConfig = LazyLoad("langroid.language_models.azure_openai.AzureConfig")
    AzureGPT = LazyLoad("langroid.language_models.azure_openai.AzureGPT")

__all__ = [
    "utils",
    "config",
    "base",
    "openai_gpt",
    "azure_openai",
    "prompt_formatter",
    "LLMConfig",
    "LLMMessage",
    "LLMFunctionCall",
    "LLMFunctionSpec",
    "Role",
    "LLMTokenUsage",
    "LLMResponse",
    "OpenAIChatModel",
    "OpenAICompletionModel",
    "OpenAIGPTConfig",
    "OpenAIGPT",
    "AzureConfig",
    "AzureGPT",
]
