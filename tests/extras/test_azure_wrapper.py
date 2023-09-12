from contextlib import contextmanager

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.azure_openai import AzureConfig, AzureGPT
from langroid.language_models.base import LLMMessage, Role
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig

set_global(Settings(stream=True))

cfg = AzureConfig(
    max_output_tokens=100,
    min_output_tokens=10,
    cache_config=RedisCacheConfig(fake=False),
)


@contextmanager
def reload_openai_on_azure_config(new_config):
    try:
        yield
    finally:
        if isinstance(new_config, AzureConfig):
            import importlib

            import openai

            importlib.reload(openai)


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: AzureConfig = AzureConfig(
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
def test_azure_wrapper(streaming, country, capital):
    cfg.stream = streaming
    mdl = AzureGPT(config=cfg)

    question = "What is the capital of " + country + "?"

    set_global(Settings(cache=False))
    cfg.use_chat_for_completion = True
    response = mdl.generate(prompt=question, max_tokens=10)
    assert capital in response.message
    assert not response.cached

    # actual chat mode
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assitant"),
        LLMMessage(role=Role.USER, content=question),
    ]
    response = mdl.chat(messages=messages, max_tokens=10)
    assert capital in response.message
    assert not response.cached

    set_global(Settings(cache=True))
    # should be from cache this time
    response = mdl.chat(messages=messages, max_tokens=10)
    assert capital in response.message
    assert response.cached


def test_chat_agent(test_settings: Settings):
    set_global(test_settings)
    cfg = _TestChatAgentConfig()
    with reload_openai_on_azure_config(cfg):
        # just testing that these don't fail
        agent = ChatAgent(cfg)
        response = agent.llm_response("what is the capital of France?")
        assert "Paris" in response.content
