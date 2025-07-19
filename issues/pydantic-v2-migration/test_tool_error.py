#!/usr/bin/env python3
"""Test script to check tool validation error messages"""

from pydantic import Field, ValidationError

from langroid.agent.base import Agent, AgentConfig
from langroid.agent.xml_tool_message import XMLToolMessage


class FileExistsTool(XMLToolMessage):
    request: str = "file_exists"
    purpose: str = "Check if a file exists"
    file_path: str = Field(..., description="Path to check")


# Create agent
agent = Agent(AgentConfig())
agent.enable_message_handling(FileExistsTool)

# Create a validation error by passing bad data
try:
    # Missing required field
    tool = FileExistsTool.model_validate({"request": "file_exists"})
except ValidationError as ve:
    # Simulate what happens in the agent code
    error_msg = agent.tool_validation_error(ve, FileExistsTool)
    print("Error message:")
    print(error_msg)
    print("\n---\n")

    # Check if it contains the tool name
    if "file_exists" in error_msg:
        print("✓ Tool name 'file_exists' found in error message")
    else:
        print("✗ Tool name 'file_exists' NOT found in error message")
        print(f"Instead found: {error_msg}")
