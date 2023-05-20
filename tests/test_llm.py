from llmagent.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from llmagent.language_models.base import LLMMessage, Role
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.configuration import Settings, set_global
import pytest

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
def test_openai_gpt(streaming, country, capital):
    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_tokens=100,
        chat_model="gpt-3.5-turbo",
        completion_model="text-davinci-003",
        cache_config=RedisCacheConfig(fake=False),
    )

    mdl = OpenAIGPT(config=cfg)

    # completion mode
    question = "What is the capital of " + country + "?"

    set_global(Settings(cache=False))
    response = mdl.generate(prompt=question, max_tokens=10)
    assert capital in response.message
    assert not response.cached

    set_global(Settings(cache=True))
    # should be from cache this time
    response = mdl.generate(prompt=question, max_tokens=10)
    assert capital in response.message
    assert response.cached

    set_global(Settings(cache=False))
    # chat mode
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
