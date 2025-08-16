from typing import Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.azure_openai import AzureConfig, AzureGPT
from langroid.language_models.base import LLMMessage, Role
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global, settings
from langroid.vector_store.base import VectorStoreConfig

set_global(Settings(stream=True))

cfg = AzureConfig(
    max_output_tokens=100,
    min_output_tokens=10,
    cache_config=RedisCacheConfig(fake=False),
    chat_model="gpt-4o",
)


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: Optional[VectorStoreConfig] = None
    llm: AzureConfig = cfg
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
    agent_cfg = _TestChatAgentConfig()
    # just testing that these don't fail
    agent = ChatAgent(agent_cfg)
    response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content


@pytest.mark.asyncio
async def test_azure_openai_async(test_settings: Settings):
    set_global(test_settings)
    llm = AzureGPT(config=cfg)
    response = await llm.achat("What is the capital of Ontario?", max_tokens=10)
    assert "Toronto" in response.message


def test_azure_config():
    # Test the AzureConfig class model_name copied into chat_model_orig
    model = "blah"
    # turn off the `chat_model` coming from test_settings in conftest.
    settings.chat_model = ""

    # test setting model_name (deprecated; use chat_model instead)
    llm_cfg = AzureConfig(model_name=model)
    assert llm_cfg.chat_model == model
    mdl = AzureGPT(llm_cfg)
    assert mdl.chat_model_orig == model
    assert mdl.config.chat_model == model

    # test setting chat_model
    llm_cfg = AzureConfig(chat_model=model)
    assert llm_cfg.chat_model == model
    mdl = AzureGPT(llm_cfg)
    assert mdl.chat_model_orig == model
    assert mdl.config.chat_model == model

    # test setting chat_model via env var
    import os

    os.environ["AZURE_OPENAI_CHAT_MODEL"] = model
    llm_cfg = AzureConfig()
    mdl = AzureGPT(llm_cfg)
    assert llm_cfg.chat_model == model
    assert mdl.chat_model_orig == model
