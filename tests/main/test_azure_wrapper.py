import pytest

from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.azure_openai import AzureConfig, AzureGPT
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAICompletionModel,
)
from langroid.utils.configuration import Settings, set_global

set_global(Settings(stream=True))


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
def test_azure_wrapper(test_settings: Settings, streaming, country, capital):
    cfg = AzureConfig(
        stream=streaming,
        max_output_tokens=100,
        min_output_tokens=10,
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        completion_model=OpenAICompletionModel.TEXT_DA_VINCI_003,
        cache_config=RedisCacheConfig(fake=False),
    )

    mdl = AzureGPT(config=cfg)

    # completion mode
    cfg.use_chat_for_completion = False
    question = "What is the capital of " + country + "?"

    set_global(Settings(cache=False))
    response = mdl.generate(prompt=question, max_tokens=20)
    assert capital in response.message
    assert not response.cached

    set_global(Settings(cache=True))
    # should be from cache this time
    response = mdl.generate(prompt=question, max_tokens=20)
    assert capital in response.message
    assert response.cached

    set_global(Settings(cache=False))
    # chat mode via `generate`,
    # i.e. use same call as for completion, but the setting below
    # actually calls `chat` under the hood
    cfg.use_chat_for_completion = True
    # check that "generate" works when "use_chat_for_completion" is True
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
