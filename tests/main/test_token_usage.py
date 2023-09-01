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


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


# Define the configurations
openai_config = OpenAIGPTConfig(
    cache_config=RedisCacheConfig(fake=False),
    chat_model=OpenAIChatModel.GPT3_5_TURBO,
    use_chat_for_completion=True,
)

azure_config = AzureConfig(
    cache_config=RedisCacheConfig(fake=False),
    use_chat_for_completion=True,
)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("config", [openai_config, azure_config])
def test_agent(config, stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig(llm=config)
    agent = ChatAgent(cfg)
    question = "What is the capital of Canada?"
    agent.llm_response_forget(question)
    assert agent.total_llm_token_usage != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token_usage

    set_global(Settings(cache=True, stream=stream))
    print("***2nd round***")
    # this convo shouldn't change the cost and tokens because `cache` is `True`
    agent.llm_response_forget(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be doubled because the question/response pair
    # is the same
    set_global(Settings(cache=False, stream=stream))
    agent.llm_response(question)
    print("***3rd round***")
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd * 2
    assert agent.total_llm_token_cost == total_cost_after_1st_rnd * 2
    importlib.reload(openai)
