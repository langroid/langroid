import os
import random
import warnings

import openai
import pytest

import langroid as lr
import langroid.language_models as lm
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import (
    AccessWarning,
    OpenAIChatModel,
    OpenAICompletionModel,
    OpenAIGPT,
    OpenAIGPTConfig,
)
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.parsing.utils import generate_random_sentences
from langroid.utils.configuration import Settings, set_global, settings

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(False, "India", "Delhi"), (True, "France", "Paris")],
)
@pytest.mark.parametrize("use_cache", [True, False])
def test_openai_gpt(test_settings: Settings, streaming, country, capital, use_cache):
    test_settings.cache = False  # cache response but don't retrieve from cache
    set_global(test_settings)

    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=True) if use_cache else None,
    )

    mdl = OpenAIGPT(config=cfg)
    question = "What is the capital of " + country + "?"
    # chat mode via `generate`,
    # i.e. use same call as for completion, but the setting below
    # actually calls `chat` under the hood
    cfg.use_chat_for_completion = True
    # check that "generate" works when "use_chat_for_completion" is True
    response = mdl.generate(prompt=question, max_tokens=800)
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
    response = mdl.chat(messages=messages, max_tokens=500)
    assert capital in response.message
    assert not response.cached

    test_settings.cache = True
    set_global(test_settings)
    # should be from cache this time, Provided config.cache_config is not None
    response = mdl.chat(messages=messages, max_tokens=500)
    assert capital in response.message
    assert response.cached == use_cache

    # pass intentional bad msg to test error handling
    messages = [
        LLMMessage(
            role=Role.FUNCTION,
            content="Hello!",
        ),
    ]

    with pytest.raises(Exception):
        _ = mdl.chat(messages=messages, max_tokens=500)


@pytest.mark.parametrize(
    "mode, max_tokens",
    [("completion", 100), ("chat", 100), ("completion", 1000), ("chat", 1000)],
)
def _test_context_length_error(test_settings: Settings, mode: str, max_tokens: int):
    """
    Test disabled, see TODO below.
    Also it takes too long since we are trying to test
    that it raises the expected error when the context length is exceeded.
    Args:
        test_settings: from conftest.py
        mode: "completion" or "chat"
        max_tokens: number of tokens to generate
    """
    set_global(test_settings)
    set_global(Settings(cache=False))

    cfg = OpenAIGPTConfig(
        stream=False,
        max_output_tokens=max_tokens,
        completion_model=OpenAICompletionModel.TEXT_DA_VINCI_003,
        cache_config=RedisCacheConfig(fake=False),
    )
    parser = Parser(config=ParsingConfig())
    llm = OpenAIGPT(config=cfg)
    context_length = (
        llm.chat_context_length() if mode == "chat" else llm.completion_context_length()
    )

    toks_per_sentence = int(parser.num_tokens(generate_random_sentences(1000)) / 1000)
    max_sentences = int(context_length * 1.5 / toks_per_sentence)
    big_message = generate_random_sentences(max_sentences + 1)
    big_message_tokens = parser.num_tokens(big_message)
    assert big_message_tokens + max_tokens > context_length
    response = None
    # TODO need to figure out what error type to expect here
    with pytest.raises(openai.BadRequestError) as e:
        if mode == "chat":
            response = llm.chat(big_message, max_tokens=max_tokens)
        else:
            response = llm.generate(prompt=big_message, max_tokens=max_tokens)

    assert response is None
    assert "context length" in str(e.value).lower()


@pytest.mark.parametrize(
    "mdl",
    [
        lm.OpenAIChatModel.GPT4o,
        lm.GeminiModel.GEMINI_2_PRO,
        "gemini/" + lm.GeminiModel.GEMINI_2_PRO.value,
    ],
)
@pytest.mark.parametrize("ctx", [16_000, None])
def test_llm_config_context_length(mdl: str, ctx: int | None):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=mdl,
        chat_context_length=ctx,  # even if wrong, use if explicitly set
    )
    mdl = lm.OpenAIGPT(config=llm_config)
    assert mdl.chat_context_length() == ctx or mdl.info().context_length


def test_model_selection(test_settings: Settings):
    set_global(test_settings)

    defaultOpenAIChatModel = lr.language_models.openai_gpt.default_openai_chat_model

    def get_response(llm):
        llm.generate(prompt="What is the capital of France?", max_tokens=50)

    def simulate_response(llm):
        llm.run_on_first_use()

    def check_warning(
        llm,
        assert_warn,
        function=get_response,
        warning_type=AccessWarning,
        catch_errors=(ImportError,),
    ):
        if assert_warn:
            with pytest.warns(expected_warning=warning_type):
                try:
                    function(llm)
                except catch_errors:
                    pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=warning_type)

                try:
                    function(llm)
                except catch_errors:
                    pass

    # Default is GPT4o; we should not generate the warning in this case
    lr.language_models.openai_gpt.default_openai_chat_model = OpenAIChatModel.GPT4_TURBO
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT3_5_TURBO))
    check_warning(llm, False)

    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, False)

    # Default is GPT3.5 (simulate GPT 4 inaccessible)
    lr.language_models.openai_gpt.default_openai_chat_model = (
        OpenAIChatModel.GPT3_5_TURBO
    )

    # No warnings generated if we specify the model explicitly
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT3_5_TURBO))
    check_warning(llm, False)

    # No warnings generated if we are using a local model
    llm = OpenAIGPT(config=OpenAIGPTConfig(api_base="localhost:8000"))
    check_warning(llm, False, function=simulate_response)
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model="local/localhost:8000"))
    check_warning(llm, False, function=simulate_response)
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model="litellm/ollama/llama"))
    check_warning(llm, False, function=simulate_response)

    # We should warn on the first usage of a model with auto-selected GPT-3.5
    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, True)

    # We should not warn on subsequent uses and models with auto-selected GPT-3.5
    check_warning(llm, False)
    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, False)

    lr.language_models.openai_gpt.default_openai_chat_model = defaultOpenAIChatModel


def test_keys():
    # Do not override the explicit settings below
    settings.chat_model = ""

    providers = [
        "vllm",
        "ollama",
        "llamacpp",
        "openai",
        "groq",
        "gemini",
        "glhf",
        "openrouter",
        "deepseek",
        "cerebras",
    ]
    key_dict = {p: f"{p.upper()}_API_KEY" for p in providers}
    key_dict["llamacpp"] = "LLAMA_API_KEY"

    for p, var in key_dict.items():
        os.environ[var] = p

    for p in providers:
        config = lm.OpenAIGPTConfig(
            chat_model=f"{p}/model",
        )

        llm = lm.OpenAIGPT(config)

        assert llm.api_key == p

        rand_key = str(random.randint(0, 10**9))
        config = lm.OpenAIGPTConfig(
            chat_model=f"{p}/model",
            api_key=rand_key,
        )

        llm = lm.OpenAIGPT(config)
        assert llm.api_key == rand_key


@pytest.mark.xfail(
    reason="LangDB may fail due to unknown flakiness",
    run=True,
    strict=False,
)
def test_llm_langdb():
    """Test that LLM access via LangDB works."""

    llm_config_langdb = lm.OpenAIGPTConfig(
        chat_model="langdb/openai/gpt-4o-mini",
    )
    llm = lm.OpenAIGPT(config=llm_config_langdb)
    result = llm.chat("what is 3+4?")
    assert "7" in result.message
