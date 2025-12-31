import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
@pytest.mark.slow
class TestUsageTracking:
    def test_usage_tracked(self):
        """Token usage is tracked correctly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        messages = [
            LLMMessage(role=Role.USER, content="Say 'test'"),
        ]

        response = llm.chat(messages, max_tokens=10)

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # Check accumulated usage in the class-level dict
        model = config.chat_model
        assert model in OpenAIResponses.usage_cost_dict
        usage_info = OpenAIResponses.usage_cost_dict[model]
        assert usage_info.prompt_tokens == response.usage.prompt_tokens
        assert usage_info.completion_tokens == response.usage.completion_tokens

    def test_usage_accumulates(self):
        """Multiple calls accumulate usage correctly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        messages1 = [
            LLMMessage(role=Role.USER, content="Say 'one'"),
        ]
        messages2 = [
            LLMMessage(role=Role.USER, content="Say 'two'"),
        ]

        response1 = llm.chat(messages1, max_tokens=10)
        response2 = llm.chat(messages2, max_tokens=10)

        # Check accumulated usage
        model = config.chat_model
        usage_info = OpenAIResponses.usage_cost_dict[model]

        total_prompt = response1.usage.prompt_tokens + response2.usage.prompt_tokens
        total_completion = (
            response1.usage.completion_tokens + response2.usage.completion_tokens
        )

        assert usage_info.prompt_tokens == total_prompt
        assert usage_info.completion_tokens == total_completion
        assert usage_info.calls == 2

    def test_cost_calculation(self):
        """Cost is calculated correctly based on model pricing."""
        from langroid.language_models.model_info import get_model_info

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        messages = [
            LLMMessage(role=Role.USER, content="Say 'test'"),
        ]

        response = llm.chat(messages, max_tokens=10)

        # Verify cost calculation
        model_info = get_model_info(config.chat_model)
        expected_cost = (response.usage.prompt_tokens / 1000.0) * (
            model_info.input_cost_per_million / 1000.0
        ) + (response.usage.completion_tokens / 1000.0) * (
            model_info.output_cost_per_million / 1000.0
        )

        assert response.usage.cost == pytest.approx(expected_cost, rel=1e-6)

        # Check accumulated cost
        model = config.chat_model
        usage_info = OpenAIResponses.usage_cost_dict[model]
        assert usage_info.cost == pytest.approx(expected_cost, rel=1e-6)

    def test_streaming_usage_tracked(self):
        """Usage is tracked correctly for streaming responses."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=True,  # Enable streaming
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        messages = [
            LLMMessage(role=Role.USER, content="Count from 1 to 3"),
        ]

        response = llm.chat(messages, max_tokens=20)

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.usage.cost > 0

        # Check accumulated usage
        model = config.chat_model
        usage_info = OpenAIResponses.usage_cost_dict[model]
        assert usage_info.prompt_tokens == response.usage.prompt_tokens
        assert usage_info.completion_tokens == response.usage.completion_tokens
        assert usage_info.cost == response.usage.cost

    def test_cached_tokens_handling(self):
        """Cached tokens are handled correctly in cost calculation."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=Role.USER, content="Say 'test'"),
        ]

        response = llm.chat(messages, max_tokens=10)

        # Verify cached_tokens field exists (may be 0 if no caching)
        assert hasattr(response.usage, "cached_tokens")
        assert response.usage.cached_tokens >= 0

        # If there are cached tokens, verify cost calculation accounts for them
        if response.usage.cached_tokens > 0:
            from langroid.language_models.model_info import get_model_info

            model_info = get_model_info(config.chat_model)

            # Cost should use cached rate for cached tokens
            non_cached_prompt = (
                response.usage.prompt_tokens - response.usage.cached_tokens
            )
            expected_cost = (
                (non_cached_prompt / 1000.0)
                * (model_info.input_cost_per_million / 1000.0)
                + (response.usage.cached_tokens / 1000.0)
                * (model_info.cached_cost_per_million / 1000.0)
                + (response.usage.completion_tokens / 1000.0)
                * (model_info.output_cost_per_million / 1000.0)
            )

            assert response.usage.cost == pytest.approx(expected_cost, rel=1e-6)

    def test_usage_summary(self):
        """Usage summary returns correct accumulated stats."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        # Make a few calls
        for i in range(3):
            messages = [
                LLMMessage(role=Role.USER, content=f"Say 'test{i}'"),
            ]
            llm.chat(messages, max_tokens=10)

        # Get usage summary
        summary = llm.usage_cost_summary()

        # Verify summary contains the model
        assert config.chat_model in summary

        # Verify it shows accumulated usage
        model = config.chat_model
        usage_info = OpenAIResponses.usage_cost_dict[model]
        assert str(usage_info.prompt_tokens) in summary
        assert str(usage_info.completion_tokens) in summary
        assert str(usage_info.calls) in summary

    def test_reset_usage(self):
        """Reset usage clears accumulated stats."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Make a call to accumulate usage
        messages = [
            LLMMessage(role=Role.USER, content="Say 'test'"),
        ]
        llm.chat(messages, max_tokens=10)

        # Verify usage was accumulated
        model = config.chat_model
        usage_before = OpenAIResponses.usage_cost_dict[model]
        assert usage_before.prompt_tokens > 0

        # Reset usage
        llm.reset_usage_cost()

        # Verify usage was reset
        usage_after = OpenAIResponses.usage_cost_dict[model]
        assert usage_after.prompt_tokens == 0
        assert usage_after.completion_tokens == 0
        assert usage_after.cost == 0.0
        assert usage_after.calls == 0

    def test_responses_api_vs_chat_completions_usage(self):
        """Usage tracking works for both Responses API and Chat Completions fallback."""
        # This test verifies usage is tracked regardless of which API is used
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Reset usage
        llm.reset_usage_cost()

        messages = [
            LLMMessage(role=Role.USER, content="Say 'hello'"),
        ]

        # Call chat (will use Responses API if available, else Chat Completions)
        response = llm.chat(messages, max_tokens=10)

        # Usage should be tracked regardless of API used
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.cost > 0

        # Verify it's accumulated
        model = config.chat_model
        usage_info = OpenAIResponses.usage_cost_dict[model]
        assert usage_info.prompt_tokens == response.usage.prompt_tokens
        assert usage_info.completion_tokens == response.usage.completion_tokens
        assert usage_info.cost == response.usage.cost
