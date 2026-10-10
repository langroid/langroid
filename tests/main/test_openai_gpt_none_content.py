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


def _content_filter_events(
    filter_names: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the Azure OpenAI stream shape from issue #658.

    The first chunk has an empty `choices` list, the second a null `content`,
    and the third a null `content` with `finish_reason == "content_filter"`.
    """
    return [
        {"choices": []},
        {"choices": [{"delta": {"content": None}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {"content": None},
                    "finish_reason": "content_filter",
                    "content_filter_results": filter_names,
                }
            ]
        },
    ]


@pytest.mark.parametrize(
    ("filter_names", "expected"),
    [
        ({"hate": {"filtered": True}}, "Cannot respond due to content filters [hate]"),
        (
            {"hate": {"filtered": True}, "jailbreak": {"filtered": True}},
            "Cannot respond due to content filters [hate, jailbreak]",
        ),
        (
            {"hate": {"filtered": False}, "jailbreak": {"filtered": False}},
            "Cannot respond due to content filters []",
        ),
    ],
    ids=["one-filter", "two-filters", "none-reported-filtered"],
)
def test_sync_content_filter_stream_yields_nonempty_message(
    filter_names: dict[str, dict[str, Any]],
    expected: str,
) -> None:
    """A content-filtered stream still produces a non-empty message."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    response, _ = model._stream_response(
        _content_filter_events(filter_names),
        chat=True,
    )

    assert response.message == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filter_names", "expected"),
    [
        ({"hate": {"filtered": True}}, "Cannot respond due to content filters [hate]"),
        (
            {"hate": {"filtered": True}, "jailbreak": {"filtered": True}},
            "Cannot respond due to content filters [hate, jailbreak]",
        ),
        (
            {"hate": {"filtered": False}, "jailbreak": {"filtered": False}},
            "Cannot respond due to content filters []",
        ),
    ],
    ids=["one-filter", "two-filters", "none-reported-filtered"],
)
async def test_async_content_filter_stream_yields_nonempty_message(
    filter_names: dict[str, dict[str, Any]],
    expected: str,
) -> None:
    """The async twin of the sync test above: same guarantee, same wording.

    Regression test for the async path losing the content-filter handling that
    the sync path already had (issue #658).
    """
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    response, _ = await model._stream_response_async(
        _AsyncEvents(_content_filter_events(filter_names)),
        chat=True,
    )

    assert response.message == expected


@pytest.mark.asyncio
async def test_async_content_filter_message_matches_sync_twin() -> None:
    """Both stream paths must agree on the content-filtered message."""
    events = _content_filter_events({"hate": {"filtered": True}})

    sync_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    sync_response, _ = sync_model._stream_response(events, chat=True)

    async_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    async_response, _ = await async_model._stream_response_async(
        _AsyncEvents(events),
        chat=True,
    )

    assert async_response.message == sync_response.message
    assert async_response.message is not None


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
