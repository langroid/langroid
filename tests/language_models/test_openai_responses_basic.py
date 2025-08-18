import os

import pytest

from langroid.language_models.base import LanguageModel


def test_can_construct_openai_responses_config():
    # Import config and class to ensure the module is wired
    from langroid.language_models.openai_responses import (
        OpenAIResponsesConfig,
    )

    model = os.getenv(
        "OPENAI_RESPONSES_TEST_MODEL",
        os.getenv("LANGROID_RESPONSES_MODEL_TEXT", "gpt-4.1"),
    )
    cfg = OpenAIResponsesConfig(chat_model=model)
    assert cfg.type == "openai_responses"
    assert isinstance(cfg.chat_model, str)


def test_factory_route_to_responses():
    from langroid.language_models.openai_responses import (
        OpenAIResponses,
        OpenAIResponsesConfig,
    )

    model = os.getenv(
        "OPENAI_RESPONSES_TEST_MODEL",
        os.getenv("LANGROID_RESPONSES_MODEL_TEXT", "gpt-4.1"),
    )
    cfg = OpenAIResponsesConfig(chat_model=model)
    llm = LanguageModel.create(cfg)
    assert llm is not None
    assert isinstance(llm, OpenAIResponses)


@pytest.mark.openai_responses
def test_simple_nonstream_chat_returns_text_and_usage():
    """Integration test using the actual API for non-stream chat."""

    if os.getenv("OPENAI_API_KEY", "") == "":
        pytest.skip("OPENAI_API_KEY not set; skipping real API test")

    from langroid.language_models.base import LLMMessage, Role
    from langroid.language_models.openai_responses import OpenAIResponsesConfig

    model = os.getenv(
        "OPENAI_RESPONSES_TEST_MODEL",
        os.getenv("LANGROID_RESPONSES_MODEL_TEXT", "gpt-4.1"),
    )
    cfg = OpenAIResponsesConfig(chat_model=model, stream=False, temperature=0.2)
    llm = LanguageModel.create(cfg)

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are helpful."),
        LLMMessage(role=Role.USER, content="Reply with the word: pong"),
    ]

    res = llm.chat(messages, max_tokens=32)
    assert isinstance(res.message, str)
    assert len(res.message) > 0
    assert res.usage is not None
    assert res.usage.total_tokens >= 0
