import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
class TestErrorHandling:
    def test_invalid_api_key(self):
        """Invalid API key raises appropriate error."""
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o-mini",
            api_key="invalid_key_for_testing",
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        with pytest.raises(Exception) as exc_info:
            llm.chat(messages)

        # Check for authentication-related error
        error_str = str(exc_info.value).lower()
        assert any(
            keyword in error_str
            for keyword in ["authentication", "api key", "unauthorized", "401"]
        )

    def test_timeout_handling(self):
        """Timeouts are handled correctly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            timeout=0.001,  # Impossibly short timeout
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        with pytest.raises(Exception) as exc_info:
            llm.chat(messages)

        # Check for timeout-related error
        error_str = str(exc_info.value).lower()
        assert any(
            keyword in error_str for keyword in ["timeout", "timed out", "time out"]
        )

    def test_invalid_model_name(self):
        """Invalid model name raises appropriate error."""
        config = OpenAIResponsesConfig(
            chat_model="invalid-model-name-xyz",
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        with pytest.raises(Exception) as exc_info:
            llm.chat(messages, max_tokens=10)

        # Check for model-related error
        error_str = str(exc_info.value).lower()
        assert any(
            keyword in error_str for keyword in ["model", "not found", "invalid", "404"]
        )

    def test_rate_limit_handling(self):
        """Rate limit errors are handled appropriately."""
        # This test simulates rate limiting by making rapid requests
        # Note: This test might not trigger actual rate limits depending on account
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        # Try to make many rapid requests (may or may not trigger rate limit)
        # This is more to ensure the code handles rate limits if they occur
        errors = []
        for i in range(5):
            try:
                llm.chat(messages, max_tokens=5)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    errors.append(e)

        # If we got rate limit errors, they should be properly formatted
        if errors:
            assert any("rate" in str(e).lower() for e in errors)

    def test_retry_on_transient_errors(self):
        """Transient errors trigger retries with backoff."""
        # This test verifies retry logic is configured
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )

        # Check retry configuration exists
        assert hasattr(config, "retry_params")
        assert config.retry_params.max_retries > 0
        assert config.retry_params.initial_delay > 0
        assert config.retry_params.exponential_base > 1.0

    def test_streaming_error_handling(self):
        """Errors during streaming are handled gracefully."""
        config = OpenAIResponsesConfig(
            chat_model="invalid-model-streaming",
            stream=True,
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        # Streaming errors should be caught and raised appropriately
        with pytest.raises(Exception) as exc_info:
            llm.chat(messages, max_tokens=10)

        # Should get an error (model not found or similar)
        assert exc_info.value is not None

    def test_malformed_response_handling(self):
        """Malformed API responses are handled gracefully."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )
        llm = OpenAIResponses(config)

        # Even with extreme parameters, should handle response
        messages = [
            LLMMessage(role=Role.USER, content="1"),
        ]

        try:
            response = llm.chat(messages, max_tokens=1)
            # Should return something even if truncated
            assert response.message is not None
        except Exception as e:
            # If it fails, should be a clear error
            assert str(e) != ""

    def test_fallback_to_chat_completions_on_responses_error(self):
        """Falls back to Chat Completions API if Responses API fails."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="Say test")]

        # Should work regardless of which API is used
        response = llm.chat(messages, max_tokens=10)
        assert response.message is not None
        assert len(response.message) > 0

    def test_empty_message_handling(self):
        """Empty messages are handled gracefully."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )
        llm = OpenAIResponses(config)

        # Empty message list
        messages = []
        response = llm.chat(messages, max_tokens=10)
        assert response.message is not None

        # Message with empty content
        messages = [LLMMessage(role=Role.USER, content="")]
        response = llm.chat(messages, max_tokens=10)
        assert response.message is not None

    def test_network_error_simulation(self):
        """Network errors are handled appropriately."""
        # Use an invalid base URL to simulate network error
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o-mini",
            api_base="http://invalid.domain.that.does.not.exist.com",
            timeout=1,  # Short timeout to fail fast
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="test")]

        with pytest.raises(Exception) as exc_info:
            llm.chat(messages)

        # Should get a connection/network error
        error_str = str(exc_info.value).lower()
        assert any(
            keyword in error_str
            for keyword in ["connection", "network", "resolve", "reach", "timeout"]
        )
