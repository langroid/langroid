#!/usr/bin/env python3
"""Test that the TaskTool ModelPrivateAttr fix works correctly."""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.task_tool import TaskTool


class SimpleTool(ToolMessage):
    """A simple test tool."""

    request: str = "simple_tool"
    purpose: str = "Test tool"
    value: int

    def handle(self) -> str:
        return f"Result: {self.value * 2}"


class MyAgent(ChatAgent):
    """Test agent."""

    pass


def test_task_tool():
    """Test TaskTool with various tool configurations."""
    print("Testing TaskTool with ModelPrivateAttr fixes...")

    # Create parent agent with tools
    parent_config = ChatAgentConfig(name="ParentAgent")
    parent_agent = MyAgent(parent_config)

    # Enable some tools
    parent_agent.enable_message(SimpleTool, use=True, handle=True)
    parent_agent.enable_message(TaskTool, use=True, handle=True)

    print(f"Parent agent tools: {parent_agent.llm_tools_usable}")

    # Create a TaskTool instance
    task_tool = TaskTool(
        system_message="You are a helpful assistant.",
        prompt="Test the simple_tool with value 5",
        tools=["ALL"],
        agent_name="TestSubAgent",
    )

    # This should work without AttributeError
    try:
        task = task_tool._set_up_task(parent_agent)
        print("✓ Successfully created sub-task with ALL tools")
        print(f"  Sub-agent tools: {task.agent.llm_tools_usable}")
    except Exception as e:
        print(f"✗ Failed to create sub-task with ALL tools: {e}")
        return False

    # Test with specific tools
    task_tool2 = TaskTool(
        system_message="You are a helpful assistant.",
        prompt="Test the simple_tool with value 10",
        tools=["simple_tool"],
        agent_name="TestSubAgent2",
    )

    try:
        task2 = task_tool2._set_up_task(parent_agent)
        print("✓ Successfully created sub-task with specific tools")
        print(f"  Sub-agent tools: {task2.agent.llm_tools_usable}")
    except Exception as e:
        print(f"✗ Failed to create sub-task with specific tools: {e}")
        return False

    # Test with no tools
    task_tool3 = TaskTool(
        system_message="You are a helpful assistant.",
        prompt="Just respond",
        tools=["NONE"],
        agent_name="TestSubAgent3",
    )

    try:
        task3 = task_tool3._set_up_task(parent_agent)
        print("✓ Successfully created sub-task with NONE tools")
        print(f"  Sub-agent tools: {task3.agent.llm_tools_usable}")
    except Exception as e:
        print(f"✗ Failed to create sub-task with NONE tools: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_task_tool()
    if success:
        print(
            "\nAll tests passed! The TaskTool ModelPrivateAttr "
            "handling is working correctly."
        )
    else:
        print("\nSome tests failed.")
