"""Tests for direct Mistral API routing through the OpenAI client."""

import json
import os
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import patch

import httpx
import pytest

from langroid.language_models.model_info import (
    MODEL_INFO,
    MistralModel,
    ModelProvider,
    get_model_info,
)
from langroid.language_models.openai_gpt import (
    MISTRAL_BASE_URL,
    OpenAIGPT,
    OpenAIGPTConfig,
)
from langroid.utils.configuration import settings

CHAT_COMPLETION_JSON: Dict[str, Any] = {
    "id": "chatcmpl-mistral-test",
    "object": "chat.completion",
    "created": 0,
    "model": "mistral-small-latest",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "mock-response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}


def _mock_clients(
    requests: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[httpx.Client, httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json=CHAT_COMPLETION_JSON)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)


def _make_llm(requests: List[Tuple[str, Dict[str, Any]]]) -> OpenAIGPT:
    return OpenAIGPT(
        OpenAIGPTConfig(
            api_key="test-mistral-key",
            chat_model="mistral/mistral-small-latest",
            stream=False,
            cache_config=None,
            use_cached_client=False,
            http_client_factory=lambda: _mock_clients(requests),
        )
    )


@pytest.fixture(autouse=True)
def clear_global_model() -> Generator[None, None, None]:
    original = settings.chat_model
    settings.chat_model = ""
    try:
        yield
    finally:
        settings.chat_model = original


def test_mistral_prefix_configures_provider() -> None:
    llm = OpenAIGPT(
        OpenAIGPTConfig(
            api_key="test-key",
            chat_model="mistral/mistral-large-latest",
        )
    )

    assert llm.is_mistral is True
    assert llm.config.chat_model == "mistral-large-latest"
    assert llm.api_base == MISTRAL_BASE_URL
    assert llm.supports_json_schema is True
    assert llm.supports_strict_tools is False


def test_mistral_model_info_and_export() -> None:
    from langroid.language_models import MistralModel as ExportedMistralModel

    assert ExportedMistralModel is MistralModel
    assert MistralModel.MISTRAL_SMALL_LATEST.value in MODEL_INFO
    info = get_model_info(MistralModel.MISTRAL_SMALL_LATEST)
    assert info.provider == ModelProvider.MISTRAL
    assert info.context_length == 256_000
    assert info.has_structured_output is True


def test_bare_mistral_alias_uses_mistral_provider() -> None:
    llm = OpenAIGPT(
        OpenAIGPTConfig(
            api_key="test-key",
            chat_model=MistralModel.MISTRAL_LARGE_LATEST,
        )
    )

    assert llm.is_mistral is True
    assert llm.api_base == MISTRAL_BASE_URL


def test_mistral_uses_environment_key() -> None:
    with patch.dict(
        os.environ,
        {"MISTRAL_API_KEY": " env-mistral-key\n"},
        clear=False,
    ):
        llm = OpenAIGPT(OpenAIGPTConfig(chat_model="mistral/mistral-small-latest"))

    assert llm.api_key == "env-mistral-key"


def test_mistral_explicit_key_and_base_take_precedence() -> None:
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-key"}, clear=False):
        llm = OpenAIGPT(
            OpenAIGPTConfig(
                api_key="explicit-key",
                api_base="https://mistral-proxy.example/v1",
                chat_model="mistral/custom-model",
            )
        )

    assert llm.api_key == "explicit-key"
    assert llm.api_base == "https://mistral-proxy.example/v1"


def test_openai_api_base_does_not_leak_into_mistral() -> None:
    with patch.dict(
        os.environ,
        {"OPENAI_API_BASE": "http://localhost:8000/v1"},
        clear=False,
    ):
        llm = OpenAIGPT(
            OpenAIGPTConfig(
                api_key="test-key",
                chat_model="mistral/mistral-small-latest",
            )
        )

    assert llm.api_base == MISTRAL_BASE_URL


def test_mistral_stream_params_match_provider_api() -> None:
    llm = OpenAIGPT(
        OpenAIGPTConfig(
            api_key="test-key",
            chat_model="mistral/mistral-small-latest",
            stream=True,
        )
    )

    original_stream = settings.stream
    settings.stream = True
    try:
        args = llm._prep_chat_completion("hello", max_tokens=5)
    finally:
        settings.stream = original_stream

    assert args["stream"] is True
    assert args["max_tokens"] == 5
    assert "max_completion_tokens" not in args
    assert "stream_options" not in args


def test_sync_chat_uses_mistral_endpoint_and_model() -> None:
    requests: List[Tuple[str, Dict[str, Any]]] = []
    response = _make_llm(requests).chat("hello", max_tokens=5)

    assert response.message == "mock-response"
    assert requests[0][0] == f"{MISTRAL_BASE_URL}/chat/completions"
    assert requests[0][1]["model"] == "mistral-small-latest"
    assert requests[0][1]["max_tokens"] == 5
    assert "max_completion_tokens" not in requests[0][1]


@pytest.mark.asyncio
async def test_async_chat_uses_mistral_endpoint_and_model() -> None:
    requests: List[Tuple[str, Dict[str, Any]]] = []
    response = await _make_llm(requests).achat("hello", max_tokens=5)

    assert response.message == "mock-response"
    assert requests[0][0] == f"{MISTRAL_BASE_URL}/chat/completions"
    assert requests[0][1]["model"] == "mistral-small-latest"
    assert requests[0][1]["max_tokens"] == 5
    assert "max_completion_tokens" not in requests[0][1]
