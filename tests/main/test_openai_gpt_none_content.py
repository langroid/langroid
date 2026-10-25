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


FILTER_NAMES = [
    {"hate": {"filtered": True}},
    {"hate": {"filtered": True}, "jailbreak": {"filtered": True}},
    {"hate": {"filtered": False}, "jailbreak": {"filtered": False}},
]
FILTER_IDS = ["one-filter", "two-filters", "none-reported-filtered"]


@pytest.mark.parametrize("filter_names", FILTER_NAMES, ids=FILTER_IDS)
def test_sync_content_filter_stream_reports_finish_reason_not_text(
    filter_names: dict[str, dict[str, Any]],
) -> None:
    """A content-filter stop is reported as `finish_reason`, not as text.

    Text made up for it would be streamed to whoever listens to the stream (a
    voice call speaks it) and stored as the assistant's turn. The cached copy
    keeps the reason, so a cache hit reports it too.
    """
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    streamed: list[str] = []
    model.config.streamer = lambda text, _kind: streamed.append(text)
    response, cached_response = model._stream_response(
        _content_filter_events(filter_names),
        chat=True,
    )

    assert response.message == ""
    assert response.finish_reason == "content_filter"
    assert streamed == []
    replayed = model._process_chat_completion_response(
        cached=True,
        response=cached_response,
    )
    assert replayed.finish_reason == "content_filter"


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_names", FILTER_NAMES, ids=FILTER_IDS)
async def test_async_content_filter_stream_reports_finish_reason_not_text(
    filter_names: dict[str, dict[str, Any]],
) -> None:
    """The async twin of the sync test above (issue #658's stream shape)."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    streamed: list[str] = []

    async def record(text: str, _kind: Any) -> None:
        streamed.append(text)

    model.config.streamer_async = record
    response, cached_response = await model._stream_response_async(
        _AsyncEvents(_content_filter_events(filter_names)),
        chat=True,
    )

    assert response.message == ""
    assert response.finish_reason == "content_filter"
    assert streamed == []
    replayed = model._process_chat_completion_response(
        cached=True,
        response=cached_response,
    )
    assert replayed.finish_reason == "content_filter"


def _content_filter_after_text_events() -> list[dict[str, Any]]:
    """A stream the content filter cut off after part of the answer."""
    return [
        {"choices": [{"delta": {"content": "Sure thing"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {"content": None},
                    "finish_reason": "content_filter",
                    "content_filter_results": {"violence": {"filtered": True}},
                }
            ]
        },
    ]


@pytest.mark.asyncio
async def test_content_filter_after_text_keeps_the_text_and_adds_nothing() -> None:
    """Both stream paths keep what was streamed before the cut, and only that."""
    events = _content_filter_after_text_events()

    sync_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    sync_response, _ = sync_model._stream_response(events, chat=True)

    async_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    async_response, _ = await async_model._stream_response_async(
        _AsyncEvents(events),
        chat=True,
    )

    assert async_response.message == sync_response.message == "Sure thing"
    assert async_response.finish_reason == sync_response.finish_reason
    assert async_response.finish_reason == "content_filter"


def test_non_stream_content_filter_reports_finish_reason() -> None:
    """The non-stream path reports the same stop the same way."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=False))
    api_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": None},
                "finish_reason": "content_filter",
                "content_filter_results": {"hate": {"filtered": True}},
            }
        ],
        "usage": {},
    }

    response = model._process_chat_completion_response(
        cached=False,
        response=api_response,
    )

    assert response.message is None
    assert response.finish_reason == "content_filter"


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


def _null_choices_events(with_usage: bool) -> list[dict[str, Any]]:
    """Build a stream whose first chunk has a null ``choices``.

    Some OpenAI-compatible endpoints emit ``{"choices": null}`` instead of an
    empty list (typically on the usage-only chunk). The remaining chunks carry
    the actual answer.
    """
    first: dict[str, Any] = {"choices": None}
    if with_usage:
        first["usage"] = {"prompt_tokens": 1, "completion_tokens": 2}
    return [
        first,
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]


@pytest.mark.parametrize("with_usage", [True, False], ids=["usage", "no-usage"])
def test_sync_null_choices_chunk_preserves_remaining_stream(
    with_usage: bool,
) -> None:
    """A null `choices` chunk must not truncate the sync stream."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    response, _ = model._stream_response(
        _null_choices_events(with_usage),
        chat=True,
    )

    assert response.message == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_usage", [True, False], ids=["usage", "no-usage"])
async def test_async_null_choices_chunk_preserves_remaining_stream(
    with_usage: bool,
) -> None:
    """Async twin of the test above: a null `choices` chunk must not truncate it.

    Without the `choices is None` guard, `len(None)` raises inside
    `_process_stream_event_async`, the `except Exception` in
    `_stream_response_async` swallows it, and the rest of the stream is lost --
    the caller silently gets an empty message.
    """
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    response, _ = await model._stream_response_async(
        _AsyncEvents(_null_choices_events(with_usage)),
        chat=True,
    )

    assert response.message == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_usage", [True, False], ids=["usage", "no-usage"])
async def test_async_null_choices_message_matches_sync_twin(
    with_usage: bool,
) -> None:
    """Both stream paths must agree when a chunk carries a null `choices`."""
    events = _null_choices_events(with_usage)

    sync_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    sync_response, _ = sync_model._stream_response(events, chat=True)

    async_model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    async_response, _ = await async_model._stream_response_async(
        _AsyncEvents(events),
        chat=True,
    )

    assert async_response.message == sync_response.message == "hi"


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (_null_choices_events(with_usage=False), "stop"),
        (_tool_call_events(None, False), "tool_calls"),
        (_function_call_events(None, False), "function_call"),
    ],
    ids=["stop", "tool-calls", "function-call"],
)
def test_stream_reports_finish_reason(
    events: list[dict[str, Any]],
    expected: str,
) -> None:
    """Every stream reports the provider's finish_reason, not just a filter stop."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))
    response, _ = model._stream_response(events, chat=True)

    assert response.finish_reason == expected


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
