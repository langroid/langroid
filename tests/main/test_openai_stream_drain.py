"""A streamed chat completion must be read to the end of its HTTP body.

`_stream_response` / `_stream_response_async` stop processing at the usage chunk
(at the finish chunk when not streaming), which is before the stream's `[DONE]`
event and the end of its chunked body. Left unread, the response is only closed
when the stream is garbage-collected, and on HTTP/1.1 that drops the underlying
connection: every streamed turn then paid a new TCP + TLS handshake and any
keep-alive pool configured via `http_client_config` never took effect
(langroid#1289, reported by @alexagr).

These tests need no LLM calls: the first three feed a synthetic event stream
straight into the stream-processing methods and assert the tail is drained; the
last one drives the real OpenAI SDK against a local HTTP/1.1 server and counts
TCP connections (openai >= 3.30 only -- older SDKs close the response at
`[DONE]` without reading on, so no consumer can keep the connection).
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterator, List

import openai
import pytest

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

# What a chat-completions stream sends with stream_options.include_usage, plus a
# trailing event standing in for the body tail the SDK consumes after `[DONE]`.
_EVENTS: List[Dict[str, Any]] = [
    {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    },
    {"choices": []},
]


def _llm(stream: bool) -> OpenAIGPT:
    """An OpenAIGPT that is never asked to make a request."""
    return OpenAIGPT(
        OpenAIGPTConfig(
            chat_model="gpt-4.1-mini",
            api_key="test-key-not-used",
            stream=stream,
            cache_config=None,
        )
    )


class _Events(Iterator[Dict[str, Any]]):
    """Synthetic stream events that record how far they were read."""

    def __init__(self, events: List[Dict[str, Any]], fail_at: int = -1) -> None:
        self._events = events
        self._fail_at = fail_at
        self.read = 0

    def __iter__(self) -> "_Events":
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.read == self._fail_at:
            raise ConnectionError("peer closed connection")
        if self.read >= len(self._events):
            raise StopIteration
        self.read += 1
        return self._events[self.read - 1]

    @property
    def exhausted(self) -> bool:
        return self.read == len(self._events)


class _AsyncEvents(AsyncIterator[Dict[str, Any]]):
    """The async twin of `_Events`."""

    def __init__(self, events: List[Dict[str, Any]], fail_at: int = -1) -> None:
        self._sync = _Events(events, fail_at)

    def __aiter__(self) -> "_AsyncEvents":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        try:
            return next(self._sync)
        except StopIteration:
            raise StopAsyncIteration from None

    @property
    def exhausted(self) -> bool:
        return self._sync.exhausted

    @property
    def read(self) -> int:
        return self._sync.read


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
def test_sync_stream_is_drained(stream: bool) -> None:
    """The events the processing loop skips are still consumed."""
    events = _Events(_EVENTS)

    response, _ = _llm(stream)._stream_response(events, chat=True)

    assert response.message == "hi"
    assert events.exhausted, f"only {events.read} of {len(_EVENTS)} events read"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
async def test_async_stream_is_drained(stream: bool) -> None:
    """The async twin of `test_sync_stream_is_drained`."""
    events = _AsyncEvents(_EVENTS)

    response, _ = await _llm(stream)._stream_response_async(events, chat=True)

    assert response.message == "hi"
    assert events.exhausted, f"only {events.read} of {len(_EVENTS)} events read"


def test_sync_failure_while_draining_keeps_the_answer() -> None:
    """A connection dropping while the tail is drained costs the connection only."""
    events = _Events(_EVENTS, fail_at=3)

    response, _ = _llm(True)._stream_response(events, chat=True)

    assert response.message == "hi"
    assert response.usage is not None
    assert response.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_failure_while_draining_keeps_the_answer() -> None:
    """The async twin of `test_sync_failure_while_draining_keeps_the_answer`."""
    events = _AsyncEvents(_EVENTS, fail_at=3)

    response, _ = await _llm(True)._stream_response_async(events, chat=True)

    assert response.message == "hi"
    assert response.usage is not None
    assert response.usage.completion_tokens == 1


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class _Server:
    """HTTP/1.1 keep-alive server that streams like the chat-completions API."""

    def __init__(self) -> None:
        self.connections = 0
        self.writers: List[asyncio.StreamWriter] = []

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        self.writers.append(writer)
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode().lower()
                length = int(head.split("content-length:")[1].split("\r\n")[0])
                await reader.readexactly(length)
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                base = {"id": "x", "object": "chat.completion.chunk", "created": 0}
                for event in _EVENTS[:3]:
                    body = json.dumps({**base, "model": "m", **event})
                    writer.write(_chunk(f"data: {body}\n\n".encode()))
                writer.write(_chunk(b"data: [DONE]\n\n"))
                await writer.drain()
                # the end of the chunked body follows [DONE], as from the API
                await asyncio.sleep(0.05)
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass


@pytest.mark.asyncio
@pytest.mark.skipif(
    tuple(int(p) for p in openai.__version__.split(".")[:2]) < (3, 30),
    reason="openai<3.30 closes a stream's response at [DONE] without reading on",
)
# Teardown noise, not the code under test: force-closing the server side of a
# connection the client still holds pooled leaves the SDK's SSE-decoder generator
# unfinalized. The same three turns through the raw SDK, with nothing force-closed,
# warn about nothing -- and so does `achat` itself (see the issue's repro script).
@pytest.mark.filterwarnings("ignore:coroutine method 'aclose' of:RuntimeWarning")
async def test_streamed_turns_reuse_one_connection() -> None:
    """End to end: three streamed turns over one pooled TCP connection."""
    server = _Server()
    listener = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        llm = OpenAIGPT(
            OpenAIGPTConfig(
                chat_model="m",
                api_base=f"http://127.0.0.1:{port}/v1",
                api_key="test-key-not-used",
                stream=True,
                use_cached_client=False,
                cache_config=None,
            )
        )
        for _ in range(3):
            response = await llm.achat("hi", max_tokens=10)
            assert response.message == "hi"
        assert server.connections == 1
    finally:
        # drop the server side of the pooled connection, else the handler stays
        # blocked reading the next request and the listener never finishes closing
        for writer in server.writers:
            writer.close()
        listener.close()
