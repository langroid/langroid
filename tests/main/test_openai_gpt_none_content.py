"""Regression tests for absent OpenAI response content."""

from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

EventFactory = Callable[[str | None, bool], list[dict[str, Any]]]


def _tool_call_events(
    content: str | None,
    include_content: bool,
) -> list[dict[str, Any]]:
    """Build a tool-call stream with optional explicit content."""
    delta: dict[str, Any] = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":"Paris"}',
                },
            }
        ]
    }
    if include_content:
        delta["content"] = content
    return [
        {"choices": [{"delta": delta, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]


def _function_call_events(
    content: str | None,
    include_content: bool,
) -> list[dict[str, Any]]:
    """Build a legacy function-call stream with optional explicit content."""
    delta: dict[str, Any] = {
        "function_call": {
            "name": "get_weather",
            "arguments": '{"location":"Paris"}',
        }
    }
    if include_content:
        delta["content"] = content
    return [
        {"choices": [{"delta": delta, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "function_call"}]},
    ]


class _AsyncEvents(AsyncIterator[dict[str, Any]]):
    """Minimal async iterator over synthetic OpenAI stream events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> "_AsyncEvents":
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.parametrize(
    "event_factory",
    [_tool_call_events, _function_call_events],
    ids=["tool-call", "function-call"],
)
@pytest.mark.parametrize(
    ("content", "include_content", "expected"),
    [(None, False, None), ("", True, "")],
    ids=["missing-content", "explicit-empty-content"],
)
def test_stream_call_content_none_vs_empty_and_cache_replay(
    event_factory: EventFactory,
    content: str | None,
    include_content: bool,
    expected: str | None,
) -> None:
    """Sync call-only streams preserve missing versus explicit empty content."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=False))
    response, cached_response = model._stream_response(
        event_factory(content, include_content),
        chat=True,
    )

    assert response.message is expected
    assert cached_response["choices"][0]["message"]["content"] is expected

    replayed = model._process_chat_completion_response(
        cached=True,
        response=cached_response,
    )
    assert replayed.message is expected
    assert replayed.cached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_factory",
    [_tool_call_events, _function_call_events],
    ids=["tool-call", "function-call"],
)
@pytest.mark.parametrize(
    ("content", "include_content", "expected"),
    [(None, False, None), ("", True, "")],
    ids=["missing-content", "explicit-empty-content"],
)
async def test_async_stream_call_content_none_vs_empty_and_cache_replay(
    event_factory: EventFactory,
    content: str | None,
    include_content: bool,
    expected: str | None,
) -> None:
    """Async call-only streams preserve missing versus explicit empty content."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=False))
    response, cached_response = await model._stream_response_async(
        _AsyncEvents(event_factory(content, include_content)),
        chat=True,
    )

    assert response.message is expected
    assert cached_response["choices"][0]["message"]["content"] is expected

    replayed = model._process_chat_completion_response(
        cached=True,
        response=cached_response,
    )
    assert replayed.message is expected
    assert replayed.cached


def test_tool_call_response_with_absent_content_stays_none() -> None:
    """A non-stream tool-call response may omit content altogether."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=False))
    api_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {},
    }

    response = model._process_chat_completion_response(
        cached=False,
        response=api_response,
    )

    assert response.message is None
    assert response.oai_tool_calls is not None
    assert response.oai_tool_calls[0].function.name == "get_weather"
