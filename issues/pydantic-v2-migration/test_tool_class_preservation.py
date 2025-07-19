#!/usr/bin/env python3
"""
Test script to verify Tool Class Preservation in ValidationErrors (Fix #3)
"""

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.pydantic_v1 import Field, ValidationError


class TestTool(ToolMessage):
    """Test tool with required fields"""

    request: str = "test_tool"
    purpose: str = "Test tool class preservation in ValidationError"
    required_field: str = Field(..., description="A required field")
    optional_field: str = Field("default", description="An optional field")

    def handle(self) -> str:
        return f"Handled {self.required_field}"


def test_tool_class_preservation():
    """Test that tool class is preserved through ValidationError handling"""

    # Create agent with mock LLM
    config = ChatAgentConfig(
        name="TestAgent",
        llm=MockLMConfig(response_dict={"content": "Test"}),
    )
    agent = ChatAgent(config)
    agent.enable_message_handling(TestTool)

    print("Testing Tool Class Preservation in ValidationErrors\n")
    print("=" * 60)

    # Test 1: Get tool messages with validation error
    print("\nTest 1: Get tool messages with validation error")
    print("-" * 40)

    # Create a message with invalid tool usage (missing required field)
    invalid_tool_msg = '{"request": "test_tool", "optional_field": "value"}'

    # Try to get tool messages - should raise ValidationError
    try:
        tools = agent.get_tool_messages(invalid_tool_msg)
        print(f"No error raised, got tools: {tools}")
    except ValidationError as ve:
        # This is what we expect
        error_msg = agent.tool_validation_error(ve)
        print("ValidationError raised as expected")
        print(f"Error message:\n{error_msg}")
        if "test_tool" in error_msg:
            print("\n✓ Tool name 'test_tool' found in error message")
        else:
            print("\n✗ Tool name 'test_tool' NOT found in error message")

    # Test 2: Direct ValidationError handling
    print("\n\nTest 2: Direct ValidationError handling")
    print("-" * 40)

    # Create a ValidationError directly
    try:
        TestTool.model_validate({"request": "test_tool"})
    except ValidationError as ve:
        # Manually attach tool_class like the agent code does
        ve.tool_class = TestTool  # type: ignore

        # Test the error message generation
        error_msg = agent.tool_validation_error(ve)
        print(f"Error message with attached tool_class:\n{error_msg}")

        if "test_tool" in error_msg:
            print("\n✓ Tool name extracted from attached tool_class")
        else:
            print("\n✗ Failed to extract tool name from attached tool_class")

    # Test 3: Check that tool_validation_error works with both parameters
    print("\n\nTest 3: tool_validation_error with explicit tool_class parameter")
    print("-" * 40)

    # Create a fresh ValidationError without tool_class attribute
    try:
        TestTool.model_validate({"request": "test_tool"})
    except ValidationError as ve_fresh:
        # First without tool_class parameter
        error_msg1 = agent.tool_validation_error(ve_fresh)
        print(f"Without tool_class param: {'test_tool' in error_msg1}")

        # Then with tool_class parameter
        error_msg2 = agent.tool_validation_error(ve_fresh, TestTool)
        print(f"With tool_class param: {'test_tool' in error_msg2}")

        if "test_tool" in error_msg2:
            print("✓ tool_class parameter works as fallback")
        else:
            print("✗ tool_class parameter fallback failed")

    print("\n" + "=" * 60)
    print("Tool Class Preservation Test Complete")


if __name__ == "__main__":
    test_tool_class_preservation()
