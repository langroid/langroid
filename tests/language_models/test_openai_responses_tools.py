import json
import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


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

        # Parse arguments to verify Paris is mentioned
        args = json.loads(response.oai_tool_calls[0].function.arguments)
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

        # Verify arguments contain Paris
        args = json.loads(response.oai_tool_calls[0].function.arguments)
        assert "paris" in args.get("query", "").lower()
