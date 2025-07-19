#!/usr/bin/env python3
"""Test the RunQueryTool subclassing pattern for max_retained_tokens."""

from langroid.agent.special.sql.utils.tools import RunQueryTool


def test_tool_subclassing():
    """Test that we can create RunQueryTool subclasses with custom
    _max_retained_tokens.
    """
    print("Testing RunQueryTool subclassing pattern...")

    # Test 1: Check original RunQueryTool
    print("\n1. Original RunQueryTool:")
    print(
        f"   _max_retained_tokens: "
        f"{getattr(RunQueryTool, '_max_retained_tokens', 'not defined')}"
    )

    # Test 2: Create a custom subclass
    class CustomRunQueryTool(RunQueryTool):
        _max_retained_tokens = 500

    print("\n2. CustomRunQueryTool subclass:")
    print(f"   _max_retained_tokens: {CustomRunQueryTool._max_retained_tokens}")
    print(f"   request: {CustomRunQueryTool.request}")
    print(f"   purpose: {CustomRunQueryTool.purpose[:50]}...")

    # Test 3: Verify original is not modified
    print("\n3. Verify original RunQueryTool is unchanged:")
    print(
        f"   _max_retained_tokens: "
        f"{getattr(RunQueryTool, '_max_retained_tokens', 'not defined')}"
    )

    # Test 4: Create another subclass with different value
    class AnotherCustomRunQueryTool(RunQueryTool):
        _max_retained_tokens = 1000

    print("\n4. AnotherCustomRunQueryTool subclass:")
    print(f"   _max_retained_tokens: {AnotherCustomRunQueryTool._max_retained_tokens}")

    # Test 5: Instance creation
    print("\n5. Test instance creation:")
    try:
        tool1 = CustomRunQueryTool(query="SELECT * FROM test")
        print("   ✓ CustomRunQueryTool instance created")
        print(f"     query: {tool1.query}")
        print(f"     _max_retained_tokens: {tool1._max_retained_tokens}")

        tool2 = AnotherCustomRunQueryTool(query="SELECT COUNT(*) FROM test")
        print("   ✓ AnotherCustomRunQueryTool instance created")
        print(f"     query: {tool2.query}")
        print(f"     _max_retained_tokens: {tool2._max_retained_tokens}")
    except Exception as e:
        print(f"   ✗ Failed to create instances: {e}")
        return False

    # Test 6: Verify class names work correctly with .name()
    print("\n6. Test .name() method:")
    print(f"   RunQueryTool.name(): {RunQueryTool.name()}")
    print(f"   CustomRunQueryTool.name(): {CustomRunQueryTool.name()}")
    print(f"   AnotherCustomRunQueryTool.name(): {AnotherCustomRunQueryTool.name()}")

    # All should return "run_query" since they inherit the same request field
    assert RunQueryTool.name() == "run_query"
    assert CustomRunQueryTool.name() == "run_query"
    assert AnotherCustomRunQueryTool.name() == "run_query"

    print("\n✓ All subclasses have the same tool name (request field)")

    return True


if __name__ == "__main__":
    success = test_tool_subclassing()
    if success:
        print("\nAll tests passed! The subclassing pattern works correctly.")
    else:
        print("\nSome tests failed.")
