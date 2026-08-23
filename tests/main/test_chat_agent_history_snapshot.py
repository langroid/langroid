"""Tests for portable ChatAgent message-history snapshots."""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models import LLMResponse
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMMessage,
    OpenAIToolCall,
    Role,
)
from langroid.language_models.mock_lm import MockLM, MockLMConfig
from langroid.parsing.file_attachment import FileAttachment


class _HistoryCheckingMockLM(MockLM):
    """MockLM that verifies restored turns reach the model chat call."""

    saw_restored_history: bool = False

    def chat(
        self,
        messages: str | list[LLMMessage],
        *args: Any,
        **kwargs: Any,
    ) -> LLMResponse:
        """Assert the full restored dialog is passed through."""
        assert isinstance(messages, list)
        assert [message.content for message in messages] == [
            "fresh system",
            "first question",
            "first answer",
            "follow-up",
        ]
        self.saw_restored_history = True
        return super().chat(messages, *args, **kwargs)


class _NeverRunHistoryTool(ToolMessage):
    """Tool whose handler records any accidental execution during import."""

    request: str = "history_never_run_sentinel"
    purpose: str = "Verify history import never executes registered tools"
    city: str
    handled_cities: ClassVar[list[str]] = []

    def handle(self) -> str:
        """Record execution so snapshot tests can reject it."""
        self.handled_cities.append(self.city)
        return self.city


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
        LLMMessage(role=Role.SYSTEM, content="snapshot system"),
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
        ),
    ]

    snapshot = agent.export_history()
    restored = ChatAgent(ChatAgentConfig(llm=None))
    restored.import_history(snapshot)

    payload = json.loads(snapshot)
    assert payload["version"] == 1
    assert payload["messages"][1]["chat_document_id"] == ""
    assert payload["messages"][1]["files"][0]["content_base64"] == "AP9iaW5hcnk="
    assert agent.message_history[1].chat_document_id == "process-local-id"
    message = restored.message_history[1]
    assert message.role == Role.USER
    assert message.content == "inspect this"
    assert message.timestamp == timestamp
    assert message.chat_document_id == ""
    assert message.files[0].content == b"\x00\xffbinary"
    assert message.files[0].filename == "sample.bin"


def test_history_snapshot_rebuilds_pending_tool_calls() -> None:
    """Import derives unresolved calls from calls minus matching tool results."""
    _NeverRunHistoryTool.handled_cities.clear()
    completed = _tool_call("call-complete", "Paris")
    pending = _tool_call("call-pending", "Tokyo")
    for call in (completed, pending):
        assert call.function is not None
        call.function.name = _NeverRunHistoryTool.name()
    source = ChatAgent(ChatAgentConfig(llm=None))
    source.message_history = [
        LLMMessage(role=Role.SYSTEM, content="snapshot system"),
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
    restored.enable_message(_NeverRunHistoryTool)
    restored.import_history(source.export_history())

    assert [call.id for call in restored.oai_tool_calls] == ["call-pending"]
    assert set(restored.oai_tool_id2call) == {
        "call-complete",
        "call-pending",
    }
    assert _NeverRunHistoryTool.handled_cities == []


def test_history_snapshot_excludes_agent_config_and_registered_tools() -> None:
    """Snapshots contain messages, not agent configuration or registrations."""
    agent = ChatAgent(
        ChatAgentConfig(
            name="snapshot-config-name-sentinel",
            handle_llm_no_tool="snapshot-config-handler-sentinel",
            llm=None,
        )
    )
    agent.enable_message(_NeverRunHistoryTool)

    snapshot = agent.export_history()
    payload = json.loads(snapshot)

    assert set(payload) == {"version", "messages"}
    assert "snapshot-config-name-sentinel" not in snapshot
    assert "snapshot-config-handler-sentinel" not in snapshot
    assert _NeverRunHistoryTool.name() not in snapshot


def test_history_snapshot_restores_into_fresh_agent_and_continues() -> None:
    """A fresh agent sends restored turns to its MockLM on continuation."""
    source = ChatAgent(
        ChatAgentConfig(
            system_message="source system",
            llm=MockLMConfig(response_dict={"first question": "first answer"}),
        )
    )
    first_response = source.llm_response("first question")
    assert first_response is not None
    snapshot = source.export_history()

    mock_config = MockLMConfig(response_dict={"follow-up": "continued answer"})
    restored = ChatAgent(
        ChatAgentConfig(
            system_message="fresh system",
            llm=mock_config,
        )
    )
    checking_llm = _HistoryCheckingMockLM(mock_config)
    restored.llm = checking_llm
    restored.import_history(snapshot)

    response = restored.llm_response("follow-up")

    assert response is not None
    assert response.content == "continued answer"
    assert checking_llm.saw_restored_history
    assert restored.message_history[0].content == "fresh system"


def test_history_snapshot_rejects_non_system_first_message_atomically() -> None:
    """Imported nonempty histories must begin with a system message."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = list(agent.message_history)
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [{"role": "user", "content": "wrong first role"}],
        }
    )

    with pytest.raises(ValueError, match="first message must have role 'system'"):
        agent.import_history(snapshot)

    assert agent.message_history == original


def test_history_snapshot_rejects_empty_message_list_atomically() -> None:
    """Imported histories must contain a system message."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = list(agent.message_history)
    snapshot = json.dumps({"version": 1, "messages": []})

    with pytest.raises(ValueError, match="messages must not be empty"):
        agent.import_history(snapshot)

    assert agent.message_history == original


def test_invalid_snapshot_preserves_registered_chat_documents() -> None:
    """Failed import keeps existing history documents registered."""
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(default_response="answer")))
    response = agent.llm_response("question")
    assert response is not None
    original_history = list(agent.message_history)
    original_calls = list(agent.oai_tool_calls)
    original_call_map = dict(agent.oai_tool_id2call)
    original_ids = {
        message.chat_document_id
        for message in original_history
        if message.chat_document_id
    }
    assert original_ids
    assert all(
        ChatDocument.from_id(document_id) is not None for document_id in original_ids
    )
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "replacement"},
                {"role": "user", "content": "valid so far"},
                {
                    "role": "function",
                    "name": " ",
                    "content": "invalid blank function name",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="function result name must be nonblank"):
        agent.import_history(snapshot)

    assert agent.message_history == original_history
    assert agent.oai_tool_calls == original_calls
    assert agent.oai_tool_id2call == original_call_map
    assert all(
        ChatDocument.from_id(document_id) is not None for document_id in original_ids
    )


def test_history_snapshot_limits_total_decoded_attachment_size() -> None:
    """Attachment bytes beyond the configured import limit are rejected."""
    source = ChatAgent(ChatAgentConfig(llm=None))
    source.message_history = [
        LLMMessage(role=Role.SYSTEM, content="snapshot system"),
        LLMMessage(
            role=Role.USER,
            content="files",
            files=[
                FileAttachment(content=b"1234", filename="first.bin"),
                FileAttachment(content=b"5678", filename="second.bin"),
            ],
        ),
    ]
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = list(agent.message_history)

    with pytest.raises(ValueError, match="attachments exceed max_size_bytes=7"):
        agent.import_history(source.export_history(), max_size_bytes=7)

    assert agent.message_history == original


def test_history_snapshot_attachment_limit_defaults_to_100_mib() -> None:
    """The public import limit defaults to 100 MiB."""
    parameter = inspect.signature(ChatAgent.import_history).parameters["max_size_bytes"]

    assert parameter.default == 100 * 1024 * 1024


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            {"role": "function", "content": "result"},
            "function result name must be nonblank and normalized",
        ),
        (
            {"role": "function", "content": "result", "name": " padded "},
            "function result name must be nonblank and normalized",
        ),
        (
            {
                "role": "user",
                "content": "bad call",
                "function_call": {"name": "weather", "arguments": {}},
            },
            "function calls must have role 'assistant'",
        ),
        (
            {"role": "assistant", "content": "bad ID", "tool_call_id": "call-1"},
            "tool_call_id is only valid on tool results",
        ),
        (
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call-1",
                "function_call": {"name": "weather", "arguments": {}},
            },
            "result messages must not contain call payloads",
        ),
        (
            {
                "role": "function",
                "content": "result",
                "name": "weather",
                "tool_calls": [
                    {
                        "id": "nested-call",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {}},
                    }
                ],
            },
            "result messages must not contain call payloads",
        ),
    ],
)
def test_history_snapshot_rejects_role_field_violations_atomically(
    message: dict[str, Any],
    error: str,
) -> None:
    """Role-dependent message fields are validated before state changes."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = LLMMessage(role=Role.SYSTEM, content="keep me")
    original_call = _tool_call("existing-call", "Rome")
    agent.message_history = [original]
    agent.oai_tool_calls = [original_call]
    agent.oai_tool_id2call = {"existing-call": original_call}
    messages: list[dict[str, Any]] = [{"role": "system", "content": "snapshot system"}]
    if message["role"] == "tool":
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {}},
                    }
                ],
            }
        )
    messages.append(message)

    with pytest.raises(ValueError, match=error):
        agent.import_history(json.dumps({"version": 1, "messages": messages}))

    assert agent.message_history == [original]
    assert agent.oai_tool_calls == [original_call]
    assert agent.oai_tool_id2call == {"existing-call": original_call}


@pytest.mark.parametrize(
    "role,error",
    [
        ("system", "tool calls must have role 'assistant'"),
        ("user", "tool calls must have role 'assistant'"),
        ("function", "result messages must not contain call payloads"),
        ("tool", "result messages must not contain call payloads"),
    ],
)
def test_history_snapshot_empty_tool_calls_still_enforce_role(
    role: str,
    error: str,
) -> None:
    """An empty tool-call field is still invalid on non-assistant roles."""
    message: dict[str, Any] = {
        "role": role,
        "content": "invalid",
        "tool_calls": [],
    }
    if role == "function":
        message["name"] = "weather"
    if role == "tool":
        message["tool_call_id"] = "call-1"
    messages = [message]
    if role != "system":
        messages.insert(0, {"role": "system", "content": "snapshot system"})
    snapshot = json.dumps({"version": 1, "messages": messages})

    with pytest.raises(ValueError, match=error):
        ChatAgent(ChatAgentConfig(llm=None)).import_history(snapshot)


def test_history_snapshot_normalizes_empty_tool_calls() -> None:
    """Imported empty tool-call lists do not reach provider payloads."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "snapshot system"},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ],
        }
    )

    agent.import_history(snapshot)

    restored = agent.message_history[1]
    assert restored.tool_calls is None
    assert "tool_calls" not in restored.api_dict("test-model")


@pytest.mark.parametrize(
    "call_payload",
    [
        {"function_call": {"name": " ", "arguments": {}}},
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "", "arguments": {}},
                }
            ]
        },
    ],
)
def test_history_snapshot_rejects_blank_call_names(
    call_payload: dict[str, Any],
) -> None:
    """Assistant function names must contain non-whitespace characters."""
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "snapshot system"},
                {"role": "assistant", **call_payload},
            ],
        }
    )

    with pytest.raises(ValueError, match="function call names must be nonblank"):
        ChatAgent(ChatAgentConfig(llm=None)).import_history(snapshot)


@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity", "1e100000"],
)
def test_history_snapshot_rejects_non_finite_nested_values_atomically(
    constant: str,
) -> None:
    """Import rejects non-JSON numeric constants nested in call arguments."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    original = list(agent.message_history)
    snapshot = (
        '{"version": 1, "messages": ['
        '{"role": "system", "content": "snapshot system"},'
        '{"role": "assistant", "function_call": '
        '{"name": "weather", "arguments": {"temperature": CONSTANT}}}]}'
    ).replace("CONSTANT", constant)

    with pytest.raises(ValueError, match="history snapshot"):
        agent.import_history(snapshot)

    assert agent.message_history == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_history_snapshot_rejects_non_finite_nested_values_on_export(
    value: float,
) -> None:
    """Export rejects values that portable JSON cannot represent."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.message_history.append(
        LLMMessage(
            role=Role.ASSISTANT,
            function_call=LLMFunctionCall(
                name="weather",
                arguments={"temperature": value},
            ),
        )
    )

    with pytest.raises(ValueError, match="JSON compliant"):
        agent.export_history()


def test_history_snapshot_escapes_surrogates_for_utf8_persistence(
    tmp_path: Path,
) -> None:
    """Hostile Unicode is escaped, writable as UTF-8, and round-trips."""
    content = "hostile surrogate: \ud800"
    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.message_history = [
        LLMMessage(role=Role.SYSTEM, content="snapshot system"),
        LLMMessage(role=Role.USER, content=content),
    ]

    snapshot = agent.export_history()
    snapshot_path = tmp_path / "history.json"
    snapshot_path.write_text(snapshot, encoding="utf-8")
    restored = ChatAgent(ChatAgentConfig(llm=None))
    restored.import_history(snapshot_path.read_text(encoding="utf-8"))

    assert "\\ud800" in snapshot
    assert restored.message_history[-1].content == content


def test_history_snapshot_accepts_unsequenced_partial_multi_call_results() -> None:
    """Import leaves call sequencing checks to the next provider request."""
    agent = ChatAgent(ChatAgentConfig(llm=None))
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "snapshot system"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": {}},
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "traffic", "arguments": {}},
                        },
                    ],
                },
                {"role": "assistant", "content": "intervening"},
                {"role": "tool", "tool_call_id": "call-2", "content": "clear"},
                {"role": "function", "name": "unmatched", "content": "result"},
            ],
        }
    )

    agent.import_history(snapshot)

    assert [call.id for call in agent.oai_tool_calls] == ["call-1"]
    assert set(agent.oai_tool_id2call) == {"call-1", "call-2"}
    assert agent.message_history[-1].name == "unmatched"


@pytest.mark.parametrize("call_ids", [[None], [""], [" padded "], ["same", "same"]])
def test_history_snapshot_rejects_invalid_tool_call_ids(
    call_ids: list[str | None],
) -> None:
    """Tool-call IDs must be unique, nonblank, and whitespace-normalized."""
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "weather", "arguments": {"city": "Rome"}},
        }
        for call_id in call_ids
    ]
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "snapshot system"},
                {"role": "assistant", "tool_calls": tool_calls},
            ],
        }
    )

    with pytest.raises(ValueError, match="tool call IDs"):
        ChatAgent(ChatAgentConfig(llm=None)).import_history(snapshot)


def test_history_snapshot_rejects_tool_call_without_function() -> None:
    """Imported tool calls must include a callable function payload."""
    snapshot = json.dumps(
        {
            "version": 1,
            "messages": [
                {"role": "system", "content": "snapshot system"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="tool calls must include a function"):
        ChatAgent(ChatAgentConfig(llm=None)).import_history(snapshot)


def test_history_snapshot_wraps_pathological_json_recursion() -> None:
    """Excessively nested JSON fails through the public ValueError contract."""
    snapshot = '{"version": 1, "messages": ' + "[" * 2_000 + "]" * 2_000 + "}"

    with pytest.raises(ValueError, match="history snapshot"):
        ChatAgent(ChatAgentConfig(llm=None)).import_history(snapshot)


def test_history_snapshot_removes_replaced_chat_documents() -> None:
    """Successful import unregisters documents linked to replaced history."""
    agent = ChatAgent(ChatAgentConfig(llm=MockLMConfig(default_response="old answer")))
    response = agent.llm_response("old question")
    assert response is not None
    old_ids = {
        message.chat_document_id
        for message in agent.message_history
        if message.chat_document_id
    }
    assert old_ids
    assert all(ChatDocument.from_id(document_id) is not None for document_id in old_ids)
    source = ChatAgent(ChatAgentConfig(llm=None))
    source.message_history = [LLMMessage(role=Role.SYSTEM, content="new history")]

    agent.import_history(source.export_history())

    assert all(ChatDocument.from_id(document_id) is None for document_id in old_ids)


@pytest.mark.parametrize(
    "snapshot",
    [
        '{"version": true, "messages": []}',
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
    original_call = _tool_call("existing-call", "Rome")
    agent.message_history = [original]
    agent.oai_tool_calls = [original_call]
    agent.oai_tool_id2call = {"existing-call": original_call}

    with pytest.raises(ValueError, match="history snapshot"):
        agent.import_history(snapshot)

    assert agent.message_history == [original]
    assert agent.oai_tool_calls == [original_call]
    assert agent.oai_tool_id2call == {"existing-call": original_call}
