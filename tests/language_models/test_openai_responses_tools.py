import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


class TestToolFormatConversion:
    """Tests for tool format conversion (unit tests, no API calls)."""

    def test_tool_result_format_conversion(self):
        """Tool results are converted to function_call_output format."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        # Create messages with a tool result
        messages = [
            LLMMessage(role=Role.USER, content="Calculate 2+2"),
            LLMMessage(
                role=Role.TOOL,
                content="4",
                tool_call_id="call_abc123",
                name="calculate",
            ),
        ]

        # Convert to input parts
        input_parts = llm._messages_to_input_parts(messages)

        # Find the function_call_output part
        tool_result_parts = [
            p for p in input_parts if p.get("type") == "function_call_output"
        ]
        assert len(tool_result_parts) == 1

        # Verify format matches Responses API spec
        result_part = tool_result_parts[0]
        assert result_part["type"] == "function_call_output"
        assert result_part["call_id"] == "call_abc123"
        assert result_part["output"] == "4"

    def test_multiple_tool_results_format(self):
        """Multiple tool results are correctly converted."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="Get weather and time"),
            LLMMessage(
                role=Role.TOOL,
                content='{"temp": 72}',
                tool_call_id="call_weather",
                name="get_weather",
            ),
            LLMMessage(
                role=Role.TOOL,
                content='{"time": "3:00 PM"}',
                tool_call_id="call_time",
                name="get_time",
            ),
        ]

        input_parts = llm._messages_to_input_parts(messages)

        tool_result_parts = [
            p for p in input_parts if p.get("type") == "function_call_output"
        ]
        assert len(tool_result_parts) == 2

        # Verify both have correct format
        call_ids = {p["call_id"] for p in tool_result_parts}
        assert "call_weather" in call_ids
        assert "call_time" in call_ids

    def test_strict_flag_nested_in_function(self):
        """Strict flag is correctly nested inside function payload."""
        from langroid.language_models.base import LLMFunctionSpec, OpenAIToolSpec

        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        # Create tool spec with strict=True at top level
        tool = OpenAIToolSpec(
            type="function",
            strict=True,
            function=LLMFunctionSpec(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            ),
        )

        # Convert the tool spec
        converted = llm._convert_tool_spec(tool)

        # Strict should NOT be at top level
        assert "strict" not in converted or converted.get("strict") is None

        # Strict should be inside function payload
        assert converted["function"]["strict"] is True

    def test_strict_flag_none_not_included(self):
        """Strict flag is not included when None."""
        from langroid.language_models.base import LLMFunctionSpec, OpenAIToolSpec

        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        # Create tool spec without strict flag
        tool = OpenAIToolSpec(
            type="function",
            function=LLMFunctionSpec(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            ),
        )

        converted = llm._convert_tool_spec(tool)

        # Strict should not be present at all (or be None)
        assert "strict" not in converted
        assert "strict" not in converted.get("function", {})

    def test_assistant_messages_preserved_in_conversation(self):
        """Assistant messages are included to preserve conversation context."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        # Multi-turn conversation with assistant response
        messages = [
            LLMMessage(role=Role.USER, content="What is 2+2?"),
            LLMMessage(role=Role.ASSISTANT, content="2+2 equals 4."),
            LLMMessage(role=Role.USER, content="And what is 3+3?"),
        ]

        input_parts = llm._messages_to_input_parts(messages)

        # Should have 3 messages: user, assistant, user
        assert len(input_parts) == 3

        # First should be user message
        assert input_parts[0]["role"] == "user"
        assert input_parts[0]["content"][0]["text"] == "What is 2+2?"

        # Second should be assistant message
        assert input_parts[1]["role"] == "assistant"
        assert input_parts[1]["content"] == "2+2 equals 4."

        # Third should be user message
        assert input_parts[2]["role"] == "user"
        assert input_parts[2]["content"][0]["text"] == "And what is 3+3?"

    def test_assistant_tool_calls_preserved(self):
        """Assistant tool calls are preserved as function_call items."""
        from langroid.language_models.base import LLMFunctionCall, OpenAIToolCall

        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        # Conversation with tool call
        messages = [
            LLMMessage(role=Role.USER, content="What's the weather?"),
            LLMMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    OpenAIToolCall(
                        id="call_123",
                        type="function",
                        function=LLMFunctionCall(
                            name="get_weather",
                            arguments={"location": "NYC"},
                        ),
                    )
                ],
            ),
            LLMMessage(
                role=Role.TOOL,
                content='{"temp": 72}',
                tool_call_id="call_123",
                name="get_weather",
            ),
        ]

        input_parts = llm._messages_to_input_parts(messages)

        # Should have: user message, function_call, function_call_output
        assert len(input_parts) == 3

        # First is user
        assert input_parts[0]["role"] == "user"

        # Second is function_call (from assistant)
        assert input_parts[1]["type"] == "function_call"
        assert input_parts[1]["call_id"] == "call_123"
        assert input_parts[1]["name"] == "get_weather"
        # Arguments should be serialized as JSON string for API
        assert input_parts[1]["arguments"] == '{"location": "NYC"}'

        # Third is function_call_output (tool result)
        assert input_parts[2]["type"] == "function_call_output"
        assert input_parts[2]["call_id"] == "call_123"


@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.tools
class TestToolCalling:
    def test_tool_call_invocation(self):
        """Model correctly calls provided tool."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from langroid.language_models.base import OpenAIToolSpec

        # Define a simple weather tool
        weather_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location",
                        }
                    },
                    "required": ["location"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="What's the weather in Paris?"),
        ]

        response = llm.chat(messages, tools=[weather_tool], max_tokens=100)

        # Check that the model attempted to call the tool
        assert response.oai_tool_calls is not None
        assert len(response.oai_tool_calls) > 0
        assert response.oai_tool_calls[0].function.name == "get_weather"

        # Arguments should be a dict (parsed from JSON string)
        args = response.oai_tool_calls[0].function.arguments
        assert isinstance(args, dict)
        assert "location" in args
        assert "paris" in args["location"].lower()

    def test_tool_result_in_conversation(self):
        """Tool results are correctly processed in conversation."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from langroid.language_models.base import (
            OpenAIToolSpec,
        )

        # Define a calculator tool
        calculator_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "calculate",
                "description": "Perform a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        # First, ask a question that should trigger tool use
        messages = [
            LLMMessage(role=Role.USER, content="What is 15 * 28?"),
        ]

        response1 = llm.chat(messages, tools=[calculator_tool], max_tokens=100)

        # Verify tool was called
        assert response1.oai_tool_calls is not None
        assert len(response1.oai_tool_calls) > 0
        tool_call = response1.oai_tool_calls[0]
        assert tool_call.function.name == "calculate"

        # Simulate tool execution and add result to conversation
        messages.append(
            LLMMessage(
                role=Role.ASSISTANT,
                content=response1.message if response1.message else "",
                oai_tool_calls=response1.oai_tool_calls,
            )
        )
        messages.append(
            LLMMessage(
                role=Role.TOOL,
                content="420",  # 15 * 28 = 420
                tool_call_id=tool_call.id,
                name="calculate",
            )
        )

        # Get final response with tool result
        response2 = llm.chat(messages, tools=[calculator_tool], max_tokens=100)

        # The model should mention 420 in the response
        assert "420" in response2.message

    def test_tool_choice_none(self):
        """Tool choice 'none' prevents tool calls."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from langroid.language_models.base import OpenAIToolSpec

        weather_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="What's the weather in Tokyo?"),
        ]

        # With tool_choice="none", model should not call tools
        response = llm.chat(
            messages, tools=[weather_tool], tool_choice="none", max_tokens=100
        )

        # Should not have tool calls
        assert response.oai_tool_calls is None or len(response.oai_tool_calls) == 0
        # Should have a text message instead
        assert len(response.message) > 0

    def test_tool_choice_required(self):
        """Tool choice 'required' forces tool use."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from langroid.language_models.base import OpenAIToolSpec

        math_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="Hello, how are you?"),
        ]

        # With tool_choice="required", model must call a tool even if not needed
        response = llm.chat(
            messages, tools=[math_tool], tool_choice="required", max_tokens=100
        )

        # Should have tool calls even for non-math question
        assert response.oai_tool_calls is not None
        assert len(response.oai_tool_calls) > 0

    def test_streaming_with_tools(self):
        """Tool calls work with streaming."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from langroid.language_models.base import OpenAIToolSpec

        search_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=True,  # Enable streaming
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="Search for information about Paris"),
        ]

        response = llm.chat(messages, tools=[search_tool], max_tokens=100)

        # Should get tool calls even with streaming
        assert response.oai_tool_calls is not None
        assert len(response.oai_tool_calls) > 0
        assert response.oai_tool_calls[0].function.name == "search"

        # Arguments should be a dict (parsed from JSON string)
        args = response.oai_tool_calls[0].function.arguments
        assert isinstance(args, dict)
        assert "paris" in args.get("query", "").lower()
