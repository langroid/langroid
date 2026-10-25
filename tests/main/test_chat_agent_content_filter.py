"""How ChatAgent replies when the provider's content filter stops a response.

The LLM layer reports the stop as `finish_reason == "content_filter"` and leaves
the reply as the provider sent it. ChatAgent then gives an empty reply the
configured `content_filter_message`, so a Task does not reject it as empty and
re-send the same message until stall protection stops it (issue #658), while the
message history keeps what the provider actually returned.
"""

from typing import Any

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.base import LLMResponse
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity

NOTICE = "Cannot respond due to content filters."


def _agent(
    monkeypatch: pytest.MonkeyPatch,
    reply: LLMResponse,
    **config: Any,
) -> tuple[ChatAgent, list[int]]:
    """A ChatAgent whose LLM always returns `reply`, counting the calls."""
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(default_response="unused"), **config)
    )
    calls: list[int] = []

    def chat(*args: Any, **kwargs: Any) -> LLMResponse:
        calls.append(1)
        return reply.model_copy()

    async def achat(*args: Any, **kwargs: Any) -> LLMResponse:
        return chat()

    assert agent.llm is not None
    monkeypatch.setattr(agent.llm, "chat", chat)
    monkeypatch.setattr(agent.llm, "achat", achat)
    return agent, calls


async def _respond(agent: ChatAgent, use_async: bool) -> Any:
    if use_async:
        return await agent.llm_response_async("hi")
    return agent.llm_response("hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_filtered_reply_gets_the_notice_and_history_keeps_it_empty(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
) -> None:
    reply = LLMResponse(message="", finish_reason="content_filter")
    agent, _ = _agent(monkeypatch, reply)

    doc = await _respond(agent, use_async)

    assert agent.config.content_filter_message == NOTICE
    assert doc.content == NOTICE
    assert doc.metadata.finish_reason == "content_filter"
    # the model is not shown a reply it never gave
    assert agent.message_history[-1].content == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_no_message_leaves_the_reply_empty(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
) -> None:
    reply = LLMResponse(message="", finish_reason="content_filter")
    agent, _ = _agent(monkeypatch, reply, content_filter_message=None)

    doc = await _respond(agent, use_async)

    assert doc.content == ""
    assert doc.metadata.finish_reason == "content_filter"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            LLMResponse(message="Sure thing", finish_reason="content_filter"),
            "Sure thing",
        ),
        (LLMResponse(message="", finish_reason="stop"), ""),
        (LLMResponse(message="", finish_reason=None), ""),
    ],
    ids=["text-before-the-cut", "other-finish-reason", "no-finish-reason"],
)
def test_only_an_empty_filtered_reply_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    reply: LLMResponse,
    expected: str,
) -> None:
    agent, _ = _agent(monkeypatch, reply)

    doc = agent.llm_response("hi")

    assert doc is not None
    assert doc.content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["run", "run_async"])
async def test_task_does_not_resend_a_filtered_message(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
) -> None:
    """Issue #658: the filtered reply must end the step, not trigger a re-send."""
    reply = LLMResponse(message="", finish_reason="content_filter")
    agent, calls = _agent(monkeypatch, reply)
    task = Task(agent, interactive=False, done_if_response=[Entity.LLM])

    if use_async:
        result = await task.run_async("hi", turns=10)
    else:
        result = task.run("hi", turns=10)

    assert result is not None
    assert result.content == NOTICE
    assert len(calls) == 1
