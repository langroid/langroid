"""
Tests for LLMResponse class, particularly the tools_content() method.
"""

from langroid.language_models.base import (
    LLMFunctionCall,
    LLMResponse,
    OpenAIToolCall,
)


class TestLLMResponseToolsContent:
    """Tests for LLMResponse.tools_content() method."""

    def test_tools_content_with_no_tools(self):
        """tools_content() should return empty string when no tools are present."""
        response = LLMResponse(message="Hello, world!")
        assert response.tools_content() == ""

    def test_tools_content_with_function_call(self):
        """tools_content() should return serialized function call when present."""
        func_call = LLMFunctionCall(
            name="search",
            arguments={"query": "weather in Paris"},
        )
        response = LLMResponse(
            message="Let me search for that.", function_call=func_call
        )

        result = response.tools_content()

        assert "FUNC:" in result
        assert "search" in result
        assert "weather in Paris" in result

    def test_tools_content_with_single_tool_call(self):
        """tools_content() should return serialized tool call when present."""
        tool_call = OpenAIToolCall(
            id="call_123",
            type="function",
            function=LLMFunctionCall(
                name="get_weather",
                arguments={"location": "New York"},
            ),
        )
        response = LLMResponse(message="", oai_tool_calls=[tool_call])

        result = response.tools_content()

        assert "OAI-TOOL:" in result
        assert "get_weather" in result
        assert "New York" in result

    def test_tools_content_with_multiple_tool_calls(self):
        """tools_content() should return all tool calls joined by newlines."""
        tool_calls = [
            OpenAIToolCall(
                id="call_1",
                type="function",
                function=LLMFunctionCall(
                    name="get_weather",
                    arguments={"location": "Paris"},
                ),
            ),
            OpenAIToolCall(
                id="call_2",
                type="function",
                function=LLMFunctionCall(
                    name="get_time",
                    arguments={"timezone": "Europe/Paris"},
                ),
            ),
        ]
        response = LLMResponse(message="", oai_tool_calls=tool_calls)

        result = response.tools_content()

        assert "get_weather" in result
        assert "Paris" in result
        assert "get_time" in result
        assert "Europe/Paris" in result
        # Should be joined by newlines
        assert "\n" in result

    def test_tools_content_function_call_takes_precedence(self):
        """function_call should take precedence over oai_tool_calls."""
        func_call = LLMFunctionCall(
            name="legacy_function",
            arguments={"arg": "value"},
        )
        tool_call = OpenAIToolCall(
            id="call_123",
            type="function",
            function=LLMFunctionCall(
                name="new_tool",
                arguments={"param": "data"},
            ),
        )
        response = LLMResponse(
            message="",
            function_call=func_call,
            oai_tool_calls=[tool_call],
        )

        result = response.tools_content()

        # function_call should take precedence
        assert "legacy_function" in result
        assert "new_tool" not in result

    def test_tools_content_consistency_with_str(self):
        """tools_content() matches __str__() when tools present."""
        func_call = LLMFunctionCall(
            name="test_func",
            arguments={"key": "value"},
        )
        response = LLMResponse(message="Some text", function_call=func_call)

        # When tools are present, both should return the tool content
        assert response.tools_content() == str(response)

    def test_tools_content_differs_from_str_when_no_tools(self):
        """tools_content() returns '' while __str__() returns message when no tools."""
        response = LLMResponse(message="Hello, world!")

        assert response.tools_content() == ""
        assert str(response) == "Hello, world!"


class TestLLMResponseStr:
    """Tests for LLMResponse.__str__() to ensure consistency."""

    def test_str_returns_message_when_no_tools(self):
        """__str__() should return message when no tools are present."""
        response = LLMResponse(message="Plain text response")
        assert str(response) == "Plain text response"

    def test_str_returns_function_call_when_present(self):
        """__str__() should return function call when present."""
        func_call = LLMFunctionCall(name="my_func", arguments={"a": 1})
        response = LLMResponse(message="Ignored text", function_call=func_call)

        result = str(response)

        assert "FUNC:" in result
        assert "my_func" in result

    def test_str_returns_tool_calls_when_present(self):
        """__str__() should return tool calls when present."""
        tool_call = OpenAIToolCall(
            id="call_1",
            type="function",
            function=LLMFunctionCall(name="tool_func", arguments={}),
        )
        response = LLMResponse(message="Ignored", oai_tool_calls=[tool_call])

        result = str(response)

        assert "OAI-TOOL:" in result
        assert "tool_func" in result
