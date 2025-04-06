import asyncio

import pytest

import langroid.language_models as lm
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models import AnthropicLLM, AnthropicLLMConfig, AnthropicModel
from langroid.language_models.base import LLMMessage, PromptVariants, Role
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
@pytest.mark.parametrize(
    "streaming, year, recipient",
    [
        (False, "2015", "Eddie Murphy"),
        (True, "2017", "David Letterman"),
    ],
)
@pytest.mark.parametrize("stream_quiet", [True, False])
async def test_anthropic_async(
    test_settings, anthropic_system_config, streaming, year, recipient, stream_quiet
):
    set_global(test_settings)

    anthropic_cfg = AnthropicLLMConfig(
        stream=streaming,
        max_output_tokens=100,
        min_output_tokens=10,
        cache_config=RedisCacheConfig(fake=False),
        async_stream_quiet=stream_quiet,
        system_config=anthropic_system_config,
    )

    anthropic = AnthropicLLM(config=anthropic_cfg)

    user_question = (
        f"What year did {recipient} win the Mark Twain Prize for American Humor?"
    )

    set_global(Settings(chat_model=AnthropicModel.CLAUDE_3_5_HAIKU, cache=False))

    prompt_variants = PromptVariants(
        anthropic=[
            {"role": Role.USER, "content": user_question},
            {
                "role": Role.ASSISTANT,
                "content": f"{recipient} won the Mark Twain Award in",
            },
        ]
    )

    generate_response = await anthropic.agenerate(
        prompt="", max_tokens=50, prompt_variants=prompt_variants
    )
    assert year in generate_response.message
    assert not generate_response.cached

    messages = [LLMMessage(role=Role.USER, content=user_question)]

    chat_response = await anthropic.achat(messages=messages, max_tokens=50)
    assert year in chat_response.message
    assert not chat_response.cached

    set_global(Settings(chat_model=AnthropicModel.CLAUDE_3_5_HAIKU, cache=True))
    cached_chat_response = await anthropic.achat(messages=messages, max_tokens=50)
    assert year in cached_chat_response.message
    assert cached_chat_response.cached

    bad_messages = [
        LLMMessage(
            role=Role.USER,
            content="Time to use a function!",
        )
    ]

    with pytest.raises(Exception):
        await anthropic.achat(
            messages=bad_messages,
            max_tokens=50,
            function_call="Unsupported Function Usage",
        )


@pytest.mark.parametrize("model_class", [OpenAIGPT, AnthropicLLM])
@pytest.mark.asyncio
async def test_llm_async_concurrent(
    test_settings: Settings, anthropic_system_config, model_class
):
    if isinstance(model_class, OpenAIGPT):
        set_global(test_settings)
    else:
        set_global(Settings(chat_model=AnthropicModel.CLAUDE_3_5_HAIKU))

    if isinstance(model_class, OpenAIGPT):
        cfg = OpenAIGPTConfig(
            stream=False,  # use streaming output if enabled globally
            type="openai",
            max_output_tokens=100,
            min_output_tokens=10,
            completion_model=OpenAICompletionModel.DAVINCI,
            cache_config=RedisCacheConfig(fake=False),
        )
    else:
        cfg = AnthropicLLMConfig(
            stream=False,
            max_output_tokens=100,
            min_output_tokens=10,
            cache_config=RedisCacheConfig(fake=False),
            system_config=anthropic_system_config,
        )

    if isinstance(model_class, OpenAIGPT):
        mdl = OpenAIGPT(config=cfg)
    else:
        mdl = AnthropicLLM(config=cfg)

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


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="LangDB may fail due to unknown flakiness!",
    run=True,
    strict=False,
)
@pytest.mark.parametrize(
    "model",
    [
        "langdb/gpt-4o-mini",
        "langdb/openai/gpt-4o-mini",
        "langdb/anthropic/claude-3-haiku-20240307",
        "langdb/claude-3-haiku-20240307",
        "langdb/gemini/gemini-2.0-flash-lite",
        "langdb/gemini-2.0-flash-lite",
    ],
)
async def test_llm_langdb(model: str):
    """Test that LLM access via LangDB works."""

    llm_config_langdb = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config_langdb)
    result = await llm.achat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "openrouter/anthropic/claude-3.5-haiku-20241022:beta",
        "openrouter/mistralai/mistral-small-24b-instruct-2501:free",
        "openrouter/google/gemini-2.0-flash-lite-001",
    ],
)
async def test_llm_openrouter(model: str):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config)
    result = await llm.achat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0
