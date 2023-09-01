import importlib

import openai
import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.azure_openai import AzureConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig

set_global(Settings(stream=True, cache=False))


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


@pytest.mark.parametrize(
    "config",
    [
        AzureConfig(
            cache_config=RedisCacheConfig(fake=False),
            use_chat_for_completion=True,
        ),
        OpenAIGPTConfig(
            cache_config=RedisCacheConfig(fake=False),
            chat_model=OpenAIChatModel.GPT4,
            use_chat_for_completion=True,
        ),
    ],
)
def test_chat_agent(config):
    # set_global(test_settings)
    cfg = _TestChatAgentConfig(llm=config)
    # just testing that these don't fail
    agent = ChatAgent(cfg)
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content
    importlib.reload(openai)
