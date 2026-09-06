"""JSON argument values must survive response parsing and stream assembly."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from langroid.language_models.base import LLMResponse
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


@pytest.fixture(
    params=[
        r'{"url": "https:\/\/example.org\/path"}',
        r'{"text": "\ud83d\ude00"}',
        '{"text": "ordinary JSON"}',
        r'{"path": "C:\\work\\file.txt"}',
        '{"enabled": true, "value": null}',
    ],
    ids=["escaped-slashes", "surrogate-pair", "plain", "backslash", "bool-null"],
)
def payload(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=["function_call", "tool_calls"])
def call_kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def model() -> OpenAIGPT:
    return OpenAIGPT(
        OpenAIGPTConfig(api_key="test-key", stream=False, cache_config=None)
    )


def _call_content(
    payload: str, call_kind: str, *, stream: bool = False
) -> dict[str, Any]:
    function = {"name": "inspect_value", "arguments": payload}
    if call_kind == "function_call":
        return {"function_call": function}
    tool: dict[str, Any] = {
        "id": "call_local",
        "type": "function",
        "function": function,
    }
    if stream:
        tool["index"] = 0
    return {"tool_calls": [tool]}


def _events(payload: str, call_kind: str) -> list[dict[str, Any]]:
    # Single-character chunks split both escape sequences and surrogate pairs.
    events = [
        {
            "choices": [
                {
                    "delta": _call_content(char, call_kind, stream=True),
                    "finish_reason": None,
                }
            ]
        }
        for char in payload
    ]
    for event in events[1:]:
        delta = event["choices"][0]["delta"]
        if call_kind == "function_call":
            del delta["function_call"]["name"]
        else:
            tool = delta["tool_calls"][0]
            tool["id"] = None
            tool["type"] = None
            tool["function"]["name"] = None
    events.append({"choices": [{"delta": {}, "finish_reason": call_kind}]})
    return events


def _assert_arguments(response: LLMResponse, payload: str, call_kind: str) -> None:
    if call_kind == "function_call":
        call = response.function_call
    else:
        assert response.oai_tool_calls is not None
        assert len(response.oai_tool_calls) == 1
        call = response.oai_tool_calls[0].function
    assert call is not None
    assert call.name == "inspect_value"
    assert call.arguments == json.loads(payload)


def test_non_stream_json_arguments(
    model: OpenAIGPT, payload: str, call_kind: str
) -> None:
    response = model._process_chat_completion_response(
        cached=False,
        response={
            "choices": [{"message": _call_content(payload, call_kind)}],
            "usage": {},
        },
    )
    _assert_arguments(response, payload, call_kind)


def test_stream_json_arguments_and_cache_replay(
    model: OpenAIGPT, payload: str, call_kind: str
) -> None:
    response, cached = model._stream_response(_events(payload, call_kind), chat=True)
    _assert_arguments(response, payload, call_kind)
    replayed = model._process_chat_completion_response(cached=True, response=cached)
    _assert_arguments(replayed, payload, call_kind)


@pytest.mark.asyncio
async def test_async_stream_json_arguments_and_cache_replay(
    model: OpenAIGPT, payload: str, call_kind: str
) -> None:
    async def events() -> AsyncIterator[dict[str, Any]]:
        for event in _events(payload, call_kind):
            yield event

    response, cached = await model._stream_response_async(events(), chat=True)
    _assert_arguments(response, payload, call_kind)
    replayed = model._process_chat_completion_response(cached=True, response=cached)
    _assert_arguments(replayed, payload, call_kind)
