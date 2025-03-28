import asyncio

import pytest

from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import (
    OpenAICompletionModel,
    OpenAIGPT,
    OpenAIGPTConfig,
)
from langroid.utils.configuration import Settings, set_global

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
@pytest.mark.parametrize("stream_quiet", [True, False])
async def test_openai_gpt_async(
    test_settings: Settings,
    streaming,
    country,
    capital,
    stream_quiet,
):
    set_global(test_settings)
    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=False),
        async_stream_quiet=stream_quiet,
    )

    mdl = OpenAIGPT(config=cfg)
    question = "What is the capital of " + country + "?"

    set_global(Settings(cache=False))
    # chat mode via `generate`,
    # i.e. use same call as for completion, but the setting below
    # actually calls `achat` under the hood
    cfg.use_chat_for_completion = True
    # check that "agenerate" works
    response = await mdl.agenerate(prompt=question, max_tokens=50)
    assert capital in response.message
    assert not response.cached

    # actual chat mode
    messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="You are a serious, helpful assistant. Be very concise, not funny",
        ),
        LLMMessage(role=Role.USER, content=question),
    ]
    response = await mdl.achat(messages=messages, max_tokens=50)
    assert capital in response.message
    assert not response.cached

    set_global(Settings(cache=True))
    # should be from cache this time
    response = await mdl.achat(messages=messages, max_tokens=50)
    assert capital in response.message
    assert response.cached

    # pass intentional bad msg to test error handling
    if not test_settings.chat_model.startswith("litellm-proxy/"):
        messages = [
            LLMMessage(
                role=Role.FUNCTION,
                content="Hello!",
            ),
        ]

        with pytest.raises(Exception):
            await mdl.achat(messages=messages, max_tokens=50)


@pytest.mark.asyncio
async def test_llm_async_concurrent(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIGPTConfig(
        stream=False,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=False),
    )

    mdl = OpenAIGPT(config=cfg)
    N = 5
    questions = ["1+" + str(i) for i in range(N)]
    expected_answers = [str(i + 1) for i in range(N)]
    answers = await asyncio.gather(
        *(mdl.agenerate(prompt=question, max_tokens=50) for question in questions)
    )

    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.message for a in answers)

    answers = await asyncio.gather(
        *(mdl.achat(question, max_tokens=50) for question in questions)
    )
    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.message for a in answers)
