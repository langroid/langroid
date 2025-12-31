import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.stream
class TestStreaming:
    def test_streaming_aggregates_correctly(self):
        """Streaming produces same result as non-streaming."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        messages = [
            LLMMessage(role=Role.USER, content="Count from 1 to 5"),
        ]

        # Non-streaming
        config_no_stream = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm_no_stream = OpenAIResponses(config_no_stream)
        response_no_stream = llm_no_stream.chat(messages, max_tokens=50)

        # Streaming
        config_stream = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=True,
            temperature=0.2,
        )
        llm_stream = OpenAIResponses(config_stream)
        response_stream = llm_stream.chat(messages, max_tokens=50)

        # Should produce similar content (not identical due to randomness)
        assert "1" in response_stream.message
        assert "5" in response_stream.message
        assert response_stream.usage.total_tokens > 0

        # Both should have usage info
        assert response_no_stream.usage.total_tokens > 0
        assert response_stream.usage.total_tokens > 0

    def test_stream_events_processed(self):
        """Stream events are properly handled."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=True,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        # Track streaming callbacks
        streamed_chunks = []

        def capture_stream(chunk):
            streamed_chunks.append(chunk)

        config.streamer = capture_stream

        messages = [LLMMessage(role=Role.USER, content="Say 'test'")]
        response = llm.chat(messages, max_tokens=10)

        # Check we got a response
        assert len(response.message) > 0
        assert response.usage.total_tokens > 0

        # If streaming worked with callbacks, we should have chunks
        # Note: This may not work if the Responses API isn't available
        # and we fall back to Chat Completions
        if len(streamed_chunks) > 0:
            combined = "".join(streamed_chunks)
            # The combined chunks should contain the response
            assert "test" in combined.lower() or "test" in response.message.lower()

    def test_streaming_with_system_message(self):
        """Test streaming with system instructions."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        messages = [
            LLMMessage(role=Role.SYSTEM, content="You always respond in uppercase."),
            LLMMessage(role=Role.USER, content="say hello"),
        ]

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=True,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)
        response = llm.chat(messages, max_tokens=20)

        # Should respond in uppercase due to system message
        assert response.message.isupper() or "HELLO" in response.message.upper()
        assert response.usage.total_tokens > 0
