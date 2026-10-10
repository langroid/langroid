"""Typed tool results remain available through the agent's public conversion API."""

from typing import Any, List, Tuple, Union

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity
from langroid.utils.types import is_instance_of


class ValueTool(ToolMessage):
    request: str = "value"
    purpose: str = "Return a typed result."
    value: Any


@pytest.mark.parametrize(
    "value, output_type",
    [
        (7, int | str),
        ("result", int | str),
        ([7, "result"], list[int | str]),
        ([7, "result"], List[Union[int, str]]),
        ((7, "result"), tuple[int, str]),
        ((7, "result"), Tuple[int, str]),
        ((7, 8), tuple[int, ...]),
    ],
)
def test_agent_extracts_typed_tool_output(value: Any, output_type: Any) -> None:
    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.enable_message(ValueTool)
    document = ChatDocument(
        content="",
        tool_messages=[ValueTool(value=value)],
        metadata=ChatDocMetaData(sender=Entity.AGENT),
    )

    assert agent.from_ChatDocument(document, output_type) == value


@pytest.mark.parametrize(
    "value, output_type",
    [
        ([7, "result"], list[int | str]),
        ((7, "result"), tuple[int, str]),
        ((7, 8), tuple[int, ...]),
    ],
)
def test_agent_extracts_typed_content(value: Any, output_type: Any) -> None:
    agent = ChatAgent(ChatAgentConfig(llm=None))
    document = ChatDocument(
        content="", content_any=value, metadata=ChatDocMetaData(sender=Entity.AGENT)
    )

    assert agent.from_ChatDocument(document, output_type) == value


@pytest.mark.parametrize(
    "value, output_type",
    [
        (False, str | float),
        ([7, "result"], list[int | float]),
        ((7, 8), tuple[int, str]),
        ((7,), tuple[int, str]),
        ((7, "result", "extra"), tuple[int, str]),
        ((), tuple[int, str]),
        ((7, "result"), tuple[int, ...]),
    ],
)
def test_agent_rejects_mismatched_tool_output(value: Any, output_type: Any) -> None:
    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.enable_message(ValueTool)
    document = ChatDocument(
        content="",
        tool_messages=[ValueTool(value=value)],
        metadata=ChatDocMetaData(sender=Entity.AGENT),
    )

    assert agent.from_ChatDocument(document, output_type) is None


@pytest.mark.parametrize("output_type", [int | None, Union[int, None]])
def test_union_type_check_accepts_none(output_type: Any) -> None:
    assert is_instance_of(None, output_type)


@pytest.mark.parametrize("output_type", [tuple, Tuple, tuple[int, ...]])
def test_unrestricted_or_variadic_tuple_accepts_empty(output_type: Any) -> None:
    assert is_instance_of((), output_type)
