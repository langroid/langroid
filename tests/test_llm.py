from llmagent.language_models.openai_gpt import (
    OpenAIGPT,
    OpenAIGPTConfig,
    OpenAIChatModel,
    OpenAICompletionModel,
)
from llmagent.language_models.base import LLMMessage, Role
from llmagent.parsing.utils import generate_random_text
from llmagent.parsing.parser import ParsingConfig, Parser
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.configuration import Settings, set_global
import openai
import pytest

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
def test_openai_gpt(test_settings: Settings, streaming, country, capital):
    set_global(test_settings)
    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAICompletionModel.TEXT_DA_VINCI_003,
        cache_config=RedisCacheConfig(fake=False),
    )

    mdl = OpenAIGPT(config=cfg)

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


@pytest.mark.parametrize(
    "mode, max_tokens",
    [("completion", 100), ("chat", 100), ("completion", 1000), ("chat", 1000)],
)
def _test_context_length_error(test_settings: Settings, mode: str, max_tokens: int):
    """
    Test disabled, see TODO below.
    Args:
        test_settings:
        mode:
        max_tokens:

    Returns:

    """
    set_global(test_settings)
    set_global(Settings(cache=False))

    cfg = OpenAIGPTConfig(
        stream=False,
        max_output_tokens=max_tokens,
        chat_model=OpenAIChatModel.GPT4,
        completion_model=OpenAICompletionModel.TEXT_DA_VINCI_003,
        cache_config=RedisCacheConfig(fake=False),
    )
    parser = Parser(config=ParsingConfig())
    llm = OpenAIGPT(config=cfg)
    context_length = (
        llm.chat_context_length() if mode == "chat" else llm.completion_context_length()
    )

    toks_per_sentence = int(parser.num_tokens(generate_random_text(1000)) / 1000)
    max_sentences = int(context_length / toks_per_sentence)
    big_message = generate_random_text(max_sentences + 1)
    assert parser.num_tokens(big_message) + max_tokens > context_length
    response = None
    # TODO need to figure out what error type to expect here
    with pytest.raises(openai.error.InvalidRequestError) as e:
        if mode == "chat":
            response = llm.chat(big_message, max_tokens=max_tokens)
        else:
            response = llm.generate(prompt=big_message, max_tokens=max_tokens)

    assert response is None
    assert "context length" in str(e.value).lower()
