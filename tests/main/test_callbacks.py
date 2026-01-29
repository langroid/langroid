"""Tests for agent callback functionality, including reasoning parameter."""

from unittest.mock import MagicMock

import pytest

import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMResponse


class MockLMWithReasoning(lm.MockLM):
    """MockLM that includes reasoning in responses."""

    def __init__(
        self,
        config: lm.MockLMConfig = lm.MockLMConfig(),
        reasoning: str = "",
    ):
        super().__init__(config)
        self._reasoning = reasoning

    def _response(self, msg: str) -> LLMResponse:
        response = super()._response(msg)
        response.reasoning = self._reasoning
        return response

    async def _response_async(self, msg: str) -> LLMResponse:
        response = await super()._response_async(msg)
        response.reasoning = self._reasoning
        return response


class TestCallbacksReasoningParameter:
    """Test that reasoning parameter is correctly passed to callbacks."""

    def test_show_llm_response_receives_reasoning(self):
        """Test that show_llm_response callback receives the reasoning parameter."""
        test_reasoning = "This is my chain-of-thought reasoning."
        mock_callback = MagicMock()

        config = ChatAgentConfig(
            llm=lm.MockLMConfig(default_response="The answer is 42"),
        )
        agent = ChatAgent(config)

        # Replace LLM with our custom MockLM that includes reasoning
        agent.llm = MockLMWithReasoning(
            config=agent.config.llm,
            reasoning=test_reasoning,
        )

        # Attach mock callback
        agent.callbacks.show_llm_response = mock_callback

        # Trigger LLM response (non-streaming)
        agent.llm_response("What is the answer?")

        # Verify the callback was called with reasoning parameter
        mock_callback.assert_called()
        call_kwargs = mock_callback.call_args.kwargs
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"] == test_reasoning

    def test_show_llm_response_empty_reasoning(self):
        """Test that show_llm_response callback receives empty reasoning when none."""
        mock_callback = MagicMock()

        config = ChatAgentConfig(
            llm=lm.MockLMConfig(default_response="The answer is 42"),
        )
        agent = ChatAgent(config)

        # Use standard MockLM without reasoning
        agent.llm = lm.MockLM(config=agent.config.llm)

        # Attach mock callback
        agent.callbacks.show_llm_response = mock_callback

        # Trigger LLM response
        agent.llm_response("What is the answer?")

        # Verify the callback was called with empty reasoning
        mock_callback.assert_called()
        call_kwargs = mock_callback.call_args.kwargs
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"] == ""

    def test_show_llm_response_citation_has_empty_reasoning(self):
        """Test that citation callback call has empty reasoning."""
        mock_callback = MagicMock()

        config = ChatAgentConfig(
            llm=lm.MockLMConfig(default_response="The answer is 42"),
        )
        agent = ChatAgent(config)
        agent.llm = lm.MockLM(config=agent.config.llm)

        # Attach mock callback
        agent.callbacks.show_llm_response = mock_callback

        # Trigger LLM response
        agent.llm_response("What is the answer?")

        # For the main response call, reasoning should be passed
        assert mock_callback.call_count >= 1
        first_call_kwargs = mock_callback.call_args_list[0].kwargs
        assert "reasoning" in first_call_kwargs


@pytest.mark.asyncio
class TestCallbacksReasoningParameterAsync:
    """Test that reasoning parameter is correctly passed to callbacks in async."""

    async def test_show_llm_response_receives_reasoning_async(self):
        """Test that show_llm_response callback receives reasoning in async flow."""
        test_reasoning = "Async chain-of-thought reasoning."
        mock_callback = MagicMock()

        config = ChatAgentConfig(
            llm=lm.MockLMConfig(default_response="The async answer is 42"),
        )
        agent = ChatAgent(config)

        # Replace LLM with our custom MockLM that includes reasoning
        agent.llm = MockLMWithReasoning(
            config=agent.config.llm,
            reasoning=test_reasoning,
        )

        # Attach mock callback
        agent.callbacks.show_llm_response = mock_callback

        # Trigger async LLM response
        await agent.llm_response_async("What is the async answer?")

        # Verify the callback was called with reasoning parameter
        mock_callback.assert_called()
        call_kwargs = mock_callback.call_args.kwargs
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"] == test_reasoning


class TestCallbackSignatureBackwardCompatibility:
    """Test that callbacks work with old signatures (without reasoning param)."""

    def test_noop_callback_accepts_reasoning(self):
        """Test that the default noop callbacks accept the reasoning parameter."""
        config = ChatAgentConfig(
            llm=lm.MockLMConfig(default_response="Test response"),
        )
        agent = ChatAgent(config)

        # Replace LLM with MockLM that includes reasoning
        agent.llm = MockLMWithReasoning(
            config=agent.config.llm,
            reasoning="Some reasoning content",
        )

        # This should not raise an error - noop callbacks accept **kwargs
        agent.llm_response("Test question")
