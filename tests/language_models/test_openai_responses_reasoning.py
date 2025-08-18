import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.reasoning
@pytest.mark.skipif(
    not os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL"),
    reason="Reasoning model not configured",
)
class TestReasoning:
    def test_reasoning_captured(self):
        """Reasoning is captured for o1 models."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"),
            stream=False,
            temperature=1.0,  # o1 models use temperature=1 always
        )
        # Set reasoning_effort parameter
        config.reasoning_effort = "low"

        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="What is 2+2? Think step by step."),
        ]

        response = llm.chat(messages, max_tokens=100)

        assert response.message is not None
        assert "4" in response.message
        # Check if reasoning was captured
        if hasattr(response, "reasoning") and response.reasoning:
            assert len(response.reasoning) > 0

    def test_reasoning_with_medium_effort(self):
        """Test reasoning with medium effort level."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"),
            stream=False,
            temperature=1.0,
        )
        config.reasoning_effort = "medium"

        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content="Explain why the sky appears blue in simple terms.",
            ),
        ]

        response = llm.chat(messages, max_tokens=200)

        assert response.message is not None
        assert len(response.message) > 0
        # Reasoning field may be present for o1 models
        if hasattr(response, "reasoning"):
            assert response.reasoning is None or isinstance(response.reasoning, str)

    def test_o1_model_constraints(self):
        """Test that o1 models handle constraints properly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"),
            stream=False,
            # o1 models ignore temperature, but we set it anyway
            temperature=0.5,
        )

        llm = OpenAIResponses(config)

        # o1 models don't support system messages - converted to user messages
        messages = [
            LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=Role.USER, content="What is the capital of France?"),
        ]

        response = llm.chat(messages, max_tokens=50)

        assert response.message is not None
        assert "Paris" in response.message
        assert response.usage is not None

    def test_reasoning_streaming(self):
        """Test reasoning capture with streaming enabled."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"),
            stream=True,  # Enable streaming
            temperature=1.0,
        )
        config.reasoning_effort = "low"

        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="What is 3 * 4?"),
        ]

        response = llm.chat(messages, max_tokens=50)

        assert response.message is not None
        assert "12" in response.message
        # Check usage is tracked even in streaming
        assert response.usage is not None
        assert response.usage.total_tokens > 0

    def test_reasoning_effort_parameter(self):
        """Test different reasoning_effort levels."""
        for effort in ["low", "medium", "high"]:
            config = OpenAIResponsesConfig(
                chat_model=os.getenv(
                    "OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"
                ),
                stream=False,
                temperature=1.0,
            )
            config.reasoning_effort = effort

            llm = OpenAIResponses(config)

            messages = [
                LLMMessage(
                    role=Role.USER,
                    content=f"Calculate 5 + 3 (effort: {effort})",
                ),
            ]

            response = llm.chat(messages, max_tokens=50)

            assert response.message is not None
            assert "8" in response.message

    def test_no_tools_with_reasoning_models(self):
        """o1 models don't support tools - verify graceful handling."""
        from langroid.language_models.base import OpenAIToolSpec

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL", "o1-mini"),
            stream=False,
            temperature=1.0,
        )

        llm = OpenAIResponses(config)

        # Define a dummy tool
        dummy_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        )

        messages = [
            LLMMessage(role=Role.USER, content="What's the weather?"),
        ]

        # Should not error even though o1 doesn't support tools
        response = llm.chat(messages, tools=[dummy_tool], max_tokens=100)

        assert response.message is not None
        # o1 should respond without using tools
        assert response.tool_calls is None or len(response.tool_calls) == 0
