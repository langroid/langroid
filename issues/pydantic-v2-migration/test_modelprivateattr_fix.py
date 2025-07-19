#!/usr/bin/env python3
"""Test script to verify ModelPrivateAttr handling fixes."""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1.fields import ModelPrivateAttr


class TestTool(ToolMessage):
    """Test tool for verification."""

    request: str = "test_tool"
    purpose: str = "Test tool"

    def handle(self) -> str:
        return "Tool executed"


def test_allow_llm_use():
    """Test that _allow_llm_use is handled correctly."""
    # Check if _allow_llm_use is a ModelPrivateAttr
    attr = TestTool._allow_llm_use
    print(f"Type of _allow_llm_use: {type(attr)}")
    print(f"Is ModelPrivateAttr: {isinstance(attr, ModelPrivateAttr)}")

    if isinstance(attr, ModelPrivateAttr):
        print(f"Default value: {attr.default}")
    else:
        print(f"Direct value: {attr}")


def test_max_retained_tokens():
    """Test that _max_retained_tokens is handled correctly."""
    # Check if _max_retained_tokens is a ModelPrivateAttr
    attr = TestTool._max_retained_tokens
    print(f"\nType of _max_retained_tokens: {type(attr)}")
    print(f"Is ModelPrivateAttr: {isinstance(attr, ModelPrivateAttr)}")

    if isinstance(attr, ModelPrivateAttr):
        print(f"Default value: {attr.default}")
    else:
        print(f"Direct value: {attr}")


def test_chat_agent():
    """Test ChatAgent with tools."""
    config = ChatAgentConfig()
    agent = ChatAgent(config)

    # This should work without errors now
    agent.enable_message(TestTool, use=True, handle=True)
    print("\nChatAgent successfully enabled TestTool")
    print(f"Tool is usable: {'test_tool' in agent.llm_tools_usable}")


if __name__ == "__main__":
    print("Testing ModelPrivateAttr handling fixes...\n")
    test_allow_llm_use()
    test_max_retained_tokens()
    test_chat_agent()
    print("\nAll tests completed!")
