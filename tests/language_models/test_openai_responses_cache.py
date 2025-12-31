import os
import time

import pytest

from langroid.cachedb import RedisCacheConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
@pytest.mark.cache
class TestCaching:
    def test_cache_hit(self):
        """Identical requests hit cache."""
        cache_config = RedisCacheConfig(fake=True)  # In-memory cache for testing

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=cache_config,
            stream=False,
            temperature=0,  # Deterministic for caching
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(role=Role.USER, content="Say exactly: CACHED"),
        ]

        # First call - should not hit cache
        start1 = time.time()
        response1 = llm.chat(messages, max_tokens=10)
        time1 = time.time() - start1

        # Second identical call - should hit cache
        start2 = time.time()
        response2 = llm.chat(messages, max_tokens=10)
        time2 = time.time() - start2

        # Cached response should be much faster
        assert time2 < time1 * 0.5  # At least 2x faster
        # Content should be identical
        assert response1.message == response2.message
        # Cached response should have no usage/cost
        assert response2.usage.total_tokens == 0
        assert response2.usage.cost == 0

    def test_cache_miss_on_different_messages(self):
        """Different messages don't hit cache."""
        cache_config = RedisCacheConfig(fake=True)

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=cache_config,
            stream=False,
            temperature=0,
        )
        llm = OpenAIResponses(config)

        messages1 = [LLMMessage(role=Role.USER, content="Say: ONE")]
        messages2 = [LLMMessage(role=Role.USER, content="Say: TWO")]

        response1 = llm.chat(messages1, max_tokens=10)
        response2 = llm.chat(messages2, max_tokens=10)

        # Different responses
        assert response1.message != response2.message
        # Both should have usage (no cache hit)
        assert response2.usage.total_tokens > 0

    def test_cache_miss_on_different_params(self):
        """Different parameters don't hit cache."""
        cache_config = RedisCacheConfig(fake=True)

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=cache_config,
            stream=False,
            temperature=0,
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="Say: HELLO")]

        # Different max_tokens
        llm.chat(messages, max_tokens=10)  # First call
        response2 = llm.chat(messages, max_tokens=20)

        # Second call should not hit cache (different params)
        assert response2.usage.total_tokens > 0

    def test_streaming_cache_after_completion(self):
        """Streaming responses are cached after completion."""
        cache_config = RedisCacheConfig(fake=True)

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=cache_config,
            stream=True,  # Streaming enabled
            temperature=0,
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="Say: STREAMED")]

        # First streaming call
        response1 = llm.chat(messages, max_tokens=10)

        # Switch to non-streaming for cache test
        config.stream = False
        llm2 = OpenAIResponses(config)

        # Should hit cache from streaming response
        start = time.time()
        response2 = llm2.chat(messages, max_tokens=10)
        elapsed = time.time() - start

        # Should be fast (cached)
        assert elapsed < 0.1  # Very fast cache hit
        assert response1.message == response2.message
        assert response2.usage.total_tokens == 0  # Cached

    def test_cache_with_tools(self):
        """Tool calls are properly cached."""
        from langroid.language_models.base import OpenAIToolSpec

        cache_config = RedisCacheConfig(fake=True)

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=cache_config,
            stream=False,
            temperature=0,
        )
        llm = OpenAIResponses(config)

        tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "get_time",
                "description": "Get current time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        )

        messages = [
            LLMMessage(
                role=Role.USER, content="What time is it? Use the get_time tool."
            ),
        ]

        # First call with tool
        response1 = llm.chat(messages, tools=[tool], max_tokens=50)

        # Second identical call - should hit cache
        start = time.time()
        response2 = llm.chat(messages, tools=[tool], max_tokens=50)
        elapsed = time.time() - start

        # Cache hit should be fast
        assert elapsed < 0.1
        # Tool calls should be identical
        if response1.oai_tool_calls:
            assert response2.oai_tool_calls is not None
            assert len(response1.oai_tool_calls) == len(response2.oai_tool_calls)
            for tc1, tc2 in zip(response1.oai_tool_calls, response2.oai_tool_calls):
                assert tc1.function.name == tc2.function.name

    def test_cache_disabled(self):
        """Cache can be disabled."""
        # No cache config = no caching
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            cache_config=None,  # No cache
            stream=False,
            temperature=0,
        )
        llm = OpenAIResponses(config)

        messages = [LLMMessage(role=Role.USER, content="Say: NOCACHE")]

        # Two identical calls
        response1 = llm.chat(messages, max_tokens=10)
        response2 = llm.chat(messages, max_tokens=10)

        # Both should have usage (no caching)
        assert response1.usage.total_tokens > 0
        assert response2.usage.total_tokens > 0
