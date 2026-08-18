"""Tests for portable ChatAgent message-history snapshots."""

import json
from datetime import datetime, timezone

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    OpenAIToolCall,
    Role,
)
from langroid.parsing.file_attachment import FileAttachment


def _tool_call(call_id: str, city: str) -> OpenAIToolCall:
    """Build a deterministic tool call for snapshot tests."""
    return OpenAIToolCall(
        id=call_id,
        type="function",
        function=LLMFunctionCall(
            name="weather",
            arguments={"city": city},
        ),
    )


def test_history_snapshot_round_trips_messages_and_binary_files() -> None:
    """JSON snapshots preserve message fields and arbitrary attachment bytes."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    timestamp = datetime(2026, 8, 18, 10, 30, tzinfo=timezone.utc)
    agent.message_history = [
        LLMMessage(
            role=Role.USER,
            content="inspect this",
            timestamp=timestamp,
            chat_document_id="process-local-id",
            files=[
                FileAttachment(
                    content=b"\x00\xffbinary",
                    filename="sample.bin",
                    mime_type="application/octet-stream",
                )
            ],
        )
    ]

    snapshot = agent.export_history()
    restored = ChatAgent(ChatAgentConfig(llm=None))
    restored.import_history(snapshot)

    payload = json.loads(snapshot)
    assert payload["version"] == 1
    assert payload["messages"][0]["files"][0]["content_base64"] == "AP9iaW5hcnk="
    assert agent.message_history[0].chat_document_id == "process-local-id"
    message = restored.message_history[0]
    assert message.role == Role.USER
    assert message.content == "inspect this"
    assert message.timestamp == timestamp
    assert message.chat_document_id == ""
    assert message.files[0].content == b"\x00\xffbinary"
    assert message.files[0].filename == "sample.bin"


def test_history_snapshot_rebuilds_pending_tool_calls() -> None:
    """Import derives unresolved calls from calls minus matching tool results."""
    completed = _tool_call("call-complete", "Paris")
    pending = _tool_call("call-pending", "Tokyo")
    source = ChatAgent(ChatAgentConfig(llm=None))
    source.message_history = [
        LLMMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[completed, pending],
        ),
        LLMMessage(
            role=Role.TOOL,
            content="sunny",
            tool_call_id="call-complete",
        ),
    ]

    restored = ChatAgent(ChatAgentConfig(llm=None))
    restored.import_history(source.export_history())

    assert [call.id for call in restored.oai_tool_calls] == ["call-pending"]
    assert set(restored.oai_tool_id2call) == {
        "call-complete",
        "call-pending",
    }


@pytest.mark.parametrize(
    "snapshot",
    [
        '{"version": 2, "messages": []}',
        '{"version": 1, "messages": "not-a-list"}',
        (
            '{"version": 1, "messages": [{"role": "user", '
            '"files": [{"content_base64": "%%%"}]}]}'
        ),
    ],
)
def test_invalid_history_snapshot_is_rejected_atomically(snapshot: str) -> None:
    """Invalid snapshots raise without replacing the current conversation."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = LLMMessage(role=Role.SYSTEM, content="keep me")
    agent.message_history = [original]

    with pytest.raises(ValueError, match="history snapshot"):
        agent.import_history(snapshot)

    assert agent.message_history == [original]
