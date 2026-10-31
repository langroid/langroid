"""Regression tests for langroid/langroid#1280.

When an LLM returns SEVERAL OpenAI tool-calls in one turn (which OpenAI does
by default) and one of them has an argument that fails `ToolMessage`
validation, the validation error must go back to the model as a `tool` message
for EVERY `tool_call_id`, so the model can retry -- exactly as it does when
there is a single bad call.

Before the fix, tool parsing aborted at the first bad call and the error came
back as one plain string, which (a) tripped the "multiple pending tool-calls,
but only one result!" assertion in `Agent.process_tool_results`, and (b) with
assertions off would have left the other call unanswered, which the OpenAI API
rejects on the next request.

No API key and no network are needed: only the OpenAI HTTP endpoint is faked,
via `httpx.MockTransport`.
"""

import json
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import httpx
import pytest

import langroid as lr
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import Settings, temporary_settings

# the two parallel tool-calls of the first turn: the second one is invalid
# ("four" is not an int), which is what used to crash the agent.
GOOD_CALL_ID = "call_a"
BAD_CALL_ID = "call_b"

GOOD_CALL = {
    "id": GOOD_CALL_ID,
    "type": "function",
    "function": {"name": "square", "arguments": '{"number": 3}'},
}
BAD_CALL = {
    "id": BAD_CALL_ID,
    "type": "function",
    "function": {"name": "square", "arguments": '{"number": "four"}'},
}

Handler = Callable[[httpx.Request], httpx.Response]


class SquareTool(ToolMessage):
    request: str = "square"
    purpose: str = "To compute the square of a <number>"
    number: int

    def handle(self) -> str:
        return str(self.number**2)


def _completion(message: Dict[str, Any], finish_reason: str) -> httpx.Response:
    """A minimal OpenAI chat-completion response."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def _tool_calls_then_answer(
    requests: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
) -> Handler:
    """Fake OpenAI endpoint: `calls` on the first turn, a final answer after.

    Appends each decoded request body to `requests`, so a test can inspect what
    the agent sent back after handling the tool-calls.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _completion(
                {"role": "assistant", "content": None, "tool_calls": calls},
                "tool_calls",
            )
        return _completion({"role": "assistant", "content": "9 and 16. DONE"}, "stop")

    return handler


def _make_agent(handler: Handler) -> lr.ChatAgent:
    """A ChatAgent using the OpenAI tools API against the faked endpoint."""
    clients: Tuple[httpx.Client, httpx.AsyncClient] = (
        httpx.Client(transport=httpx.MockTransport(handler)),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            use_functions_api=True,  # OpenAI tool-calls, not JSON-in-text tools
            use_tools_api=True,
            use_tools=False,
            llm=OpenAIGPTConfig(
                chat_model="gpt-4o",
                api_key="sk-test",
                stream=False,
                cache_config=None,
                use_cached_client=False,
                http_client_factory=lambda: clients,
            ),
        )
    )
    agent.enable_message(SquareTool)
    return agent


def _parallel_calls_agent(
    requests: List[Dict[str, Any]],
) -> lr.ChatAgent:
    """Agent whose first LLM turn is one good and one invalid `square` call."""
    return _make_agent(_tool_calls_then_answer(requests, [GOOD_CALL, BAD_CALL]))


def _answered_tool_call_ids(request_body: Dict[str, Any]) -> List[str]:
    """The `tool_call_id`s answered by `tool` messages in a request body."""
    return [
        m["tool_call_id"]
        for m in request_body["messages"]
        if m.get("role") == "tool" and m.get("tool_call_id") is not None
    ]


def _unparsed_llm_doc(agent: lr.ChatAgent, prompt: str) -> ChatDocument:
    """The agent's LLM response doc, with its tool-calls not yet parsed.

    Going through `llm_response` keeps the doc and the agent's pending
    tool-call bookkeeping in agreement, as they are in a real turn.
    """
    doc = agent.llm_response(prompt)
    assert doc is not None
    assert doc.oai_tool_calls is not None
    # parsing the tools is what raises the ValidationError, so the doc must
    # arrive at handle_message with its tools NOT yet extracted
    assert doc.tool_messages == []
    return doc


def test_multi_tool_call_validation_error_answers_every_call_id() -> None:
    """Sync: a bad call among parallel calls does not crash; both are answered."""
    requests: List[Dict[str, Any]] = []
    agent = _parallel_calls_agent(requests)
    task = lr.Task(agent, interactive=False, single_round=False)

    with temporary_settings(Settings(cache=False)):
        # before the fix this raised
        # AssertionError: There are multiple pending tool-calls, but only one result!
        task.run("What are the squares of 3 and 4?", turns=4)

    assert len(requests) >= 2, "the agent never sent a second request"
    assert _answered_tool_call_ids(requests[1]) == [GOOD_CALL_ID, BAD_CALL_ID]


@pytest.mark.asyncio
async def test_multi_tool_call_validation_error_answers_every_call_id_async() -> None:
    """Async: same as the sync test, on the `handle_message_async` path."""
    requests: List[Dict[str, Any]] = []
    agent = _parallel_calls_agent(requests)
    task = lr.Task(agent, interactive=False, single_round=False)

    with temporary_settings(Settings(cache=False)):
        await task.run_async("What are the squares of 3 and 4?", turns=4)

    assert len(requests) >= 2, "the agent never sent a second request"
    assert _answered_tool_call_ids(requests[1]) == [GOOD_CALL_ID, BAD_CALL_ID]


def test_handle_message_returns_error_per_tool_call_id() -> None:
    """`handle_message` maps the validation error onto every tool-call id."""
    requests: List[Dict[str, Any]] = []
    agent = _parallel_calls_agent(requests)

    with temporary_settings(Settings(cache=False)):
        doc = _unparsed_llm_doc(agent, "What are the squares of 3 and 4?")
        assert [tc.id for tc in doc.oai_tool_calls or []] == [
            GOOD_CALL_ID,
            BAD_CALL_ID,
        ]
        result = agent.handle_message(doc)

    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == [GOOD_CALL_ID, BAD_CALL_ID]
    for error in result.values():
        assert "errors in your attempt to use the" in error
        assert "square" in error


@pytest.mark.asyncio
async def test_handle_message_async_returns_error_per_tool_call_id() -> None:
    """`handle_message_async` maps the validation error onto every tool-call id."""
    requests: List[Dict[str, Any]] = []
    agent = _parallel_calls_agent(requests)

    with temporary_settings(Settings(cache=False)):
        doc = _unparsed_llm_doc(agent, "What are the squares of 3 and 4?")
        assert [tc.id for tc in doc.oai_tool_calls or []] == [
            GOOD_CALL_ID,
            BAD_CALL_ID,
        ]
        result = await agent.handle_message_async(doc)

    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == [GOOD_CALL_ID, BAD_CALL_ID]
    for error in result.values():
        assert "errors in your attempt to use the" in error
        assert "square" in error


def test_single_tool_call_validation_error_still_returns_a_string() -> None:
    """The common single-call case is unchanged: a plain error string."""
    requests: List[Dict[str, Any]] = []
    agent = _make_agent(_tool_calls_then_answer(requests, [BAD_CALL]))

    with temporary_settings(Settings(cache=False)):
        doc = _unparsed_llm_doc(agent, "What is the square of 4?")
        assert [tc.id for tc in doc.oai_tool_calls or []] == [BAD_CALL_ID]
        result = agent.handle_message(doc)

    assert isinstance(result, str)
    assert "errors in your attempt to use the" in result
