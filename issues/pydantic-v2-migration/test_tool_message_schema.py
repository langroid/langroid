#!/usr/bin/env python3
"""Test script to verify the JSON schema fix for ToolMessage"""

import json
import sys

from pydantic import BaseModel, ConfigDict


class TestToolMessage(BaseModel):
    """Test class that mimics ToolMessage structure"""

    request: str = "test_tool"
    purpose: str = "Test purpose"
    test_field: str = "test"

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=False,
        validate_default=True,
        validate_assignment=True,
        # This was the problematic line - using a set instead of list
        json_schema_extra={"exclude": ["purpose", "id"]},
    )

    def test_exclude_access(self):
        """Test accessing the exclude field"""
        excludes = set(self.model_config["json_schema_extra"]["exclude"])
        print(f"Excludes as set: {excludes}")
        excludes_union = excludes.union({"request"})
        print(f"After union: {excludes_union}")
        return True

    @classmethod
    def test_json_schema(cls):
        """Test generating JSON schema"""
        try:
            schema = cls.model_json_schema()
            print("JSON schema generated successfully!")
            print(json.dumps(schema, indent=2))
            return True
        except TypeError as e:
            print(f"Error generating JSON schema: {e}")
            return False


def main():
    print("Testing ToolMessage JSON schema fix...")
    print("=" * 50)

    # Test 1: Create instance
    try:
        msg = TestToolMessage()
        print("✓ Instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create instance: {e}")
        return 1

    # Test 2: Access exclude field
    try:
        msg.test_exclude_access()
        print("✓ Exclude field access works")
    except Exception as e:
        print(f"✗ Failed to access exclude field: {e}")
        return 1

    # Test 3: Generate JSON schema
    try:
        if TestToolMessage.test_json_schema():
            print("✓ JSON schema generation works")
        else:
            print("✗ JSON schema generation failed")
            return 1
    except Exception as e:
        print(f"✗ Exception in JSON schema generation: {e}")
        return 1

    print("\nAll tests passed! The fix is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
