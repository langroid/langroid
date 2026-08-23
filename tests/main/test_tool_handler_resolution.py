"""Tests for `_handler` resolution in tool dispatch (issue #1106).

A tool's `_handler` attribute lets a developer route a tool to a
differently-named agent method. It is declared on the tool *class*, so it
must be resolved from the class: `ToolMessage` allows extra fields, so an
LLM-supplied `"_handler"` key in tool JSON lands on the instance and, if
read from there, redirects dispatch to an arbitrary agent method.
"""

import asyncio
from typing import Optional

import pytest

import langroid as lr
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity

DANGEROUS = "DANGEROUS-HANDLER-RAN"
SAFE = "SAFE-HANDLER-RAN"
CUSTOM = "CUSTOM-HANDLER-RAN"


class SafeTool(ToolMessage):
    request: str = "safe_tool"
    purpose: str = "A benign tool"
    x: int = 1


class RedirectTool(ToolMessage):
    """Legitimate use: class-level `_handler` routes to a custom method."""

    request: str = "redirect_tool"
    purpose: str = "Routes to a custom handler name"
    x: int = 1
    _handler = "my_custom_handler"


class HandleAndRedirect(ToolMessage):
    """Has both a `handle` method and a custom `_handler` name."""

    request: str = "handle_and_redirect"
    purpose: str = "Both handle() and a custom _handler name"
    x: int = 1
    _handler = "generated_custom_name"

    def handle(self) -> str:
        return f"HANDLE-RAN x={self.x}"


class DispatchAgent(lr.ChatAgent):
    """Agent with a safe handler plus a method an attacker would target."""

    def safe_tool(self, msg: SafeTool) -> str:
        return SAFE

    async def safe_tool_async(self, msg: SafeTool) -> str:
        return SAFE

    def dangerous_method(self, msg: ToolMessage) -> str:
        return DANGEROUS

    async def dangerous_method_async(self, msg: ToolMessage) -> str:
        return DANGEROUS

    def my_custom_handler(self, msg: RedirectTool) -> str:
        return f"{CUSTOM} x={msg.x}"

    async def my_custom_handler_async(self, msg: RedirectTool) -> str:
        return f"{CUSTOM} x={msg.x}"


def _agent() -> DispatchAgent:
    agent = DispatchAgent(lr.ChatAgentConfig(llm=None, name="Dispatch"))
    agent.enable_message([SafeTool, RedirectTool])
    return agent


def _injected_json(handler: str = "dangerous_method") -> str:
    return '{"request": "safe_tool", "x": 1, "_handler": "%s"}' % handler


def _content(result: Optional[object]) -> str:
    if result is None:
        return ""
    if isinstance(result, ChatDocument):
        return result.content
    return str(result)


def test_injected_handler_does_not_redirect_dispatch() -> None:
    """An LLM-injected `_handler` key must not redirect the handler."""
    agent = _agent()
    tool = SafeTool.model_validate_json(_injected_json())
    # the injected key really is visible on the instance (extra="allow")
    assert getattr(tool, "_handler", None) == "dangerous_method"

    out = _content(agent.handle_tool_message(tool))
    assert DANGEROUS not in out
    assert SAFE in out


@pytest.mark.asyncio
async def test_injected_handler_does_not_redirect_dispatch_async() -> None:
    """Async dispatch must ignore an injected `_handler` too."""
    agent = _agent()
    tool = SafeTool.model_validate_json(_injected_json("dangerous_method"))

    out = _content(await agent.handle_tool_message_async(tool))
    assert DANGEROUS not in out
    assert SAFE in out


def test_injected_handler_blocked_via_llm_parse_path() -> None:
    """End-to-end: injected JSON arriving as LLM content is not redirected."""
    agent = _agent()
    doc = ChatDocument(
        content=_injected_json(),
        metadata=ChatDocMetaData(sender=Entity.LLM),
    )
    tools = agent.get_tool_messages(doc)
    assert [type(t).__name__ for t in tools] == ["SafeTool"]

    out = _content(agent.handle_tool_message(tools[0]))
    assert DANGEROUS not in out
    assert SAFE in out


def test_injected_handler_cannot_reach_arbitrary_agent_method() -> None:
    """Injection naming a non-tool agent method resolves to the tool's own."""
    agent = _agent()
    tool = SafeTool.model_validate_json(_injected_json("clear_history"))
    out = _content(agent.handle_tool_message(tool))
    assert SAFE in out


def test_class_level_handler_still_routes() -> None:
    """The legitimate class-level `_handler` redirect keeps working."""
    agent = _agent()
    out = _content(agent.handle_tool_message(RedirectTool(x=7)))
    assert CUSTOM in out
    assert "x=7" in out


@pytest.mark.asyncio
async def test_class_level_handler_still_routes_async() -> None:
    """The legitimate redirect works on the async path as well."""
    agent = _agent()
    out = _content(await agent.handle_tool_message_async(RedirectTool(x=9)))
    assert CUSTOM in out
    assert "x=9" in out


def test_class_level_handler_wins_over_injected_key() -> None:
    """A class-declared `_handler` is used even if JSON injects another."""
    agent = _agent()
    tool = RedirectTool.model_validate_json(
        '{"request": "redirect_tool", "x": 2, "_handler": "dangerous_method"}'
    )
    out = _content(agent.handle_tool_message(tool))
    assert DANGEROUS not in out
    assert CUSTOM in out


def test_enable_message_with_handle_and_custom_handler_name() -> None:
    """A tool with both `handle()` and `_handler` is enabled without error.

    Previously this raised
    `TypeError: attribute name must be string, not 'ModelPrivateAttr'`,
    since the class-level lookup returned pydantic's private-attr wrapper.
    """
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=None, name="EnableAgent"))
    agent.enable_message(HandleAndRedirect)
    assert hasattr(agent, "generated_custom_name")

    out = _content(agent.handle_tool_message(HandleAndRedirect(x=3)))
    assert "HANDLE-RAN" in out
    assert "x=3" in out


def test_async_injection_falls_back_to_sync_handler_safely() -> None:
    """Async dispatch falling back to a sync-only handler stays un-redirected."""

    class SyncOnlyAgent(lr.ChatAgent):
        def safe_tool(self, msg: SafeTool) -> str:
            return SAFE

        def dangerous_method(self, msg: ToolMessage) -> str:
            return DANGEROUS

    agent = SyncOnlyAgent(lr.ChatAgentConfig(llm=None, name="SyncOnly"))
    agent.enable_message(SafeTool)
    tool = SafeTool.model_validate_json(_injected_json())

    out = _content(asyncio.run(agent.handle_tool_message_async(tool)))
    assert DANGEROUS not in out
    assert SAFE in out
