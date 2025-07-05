"""
Tests for OpenAIGPT client caching functionality.
"""

import pytest

from langroid.language_models.client_cache import (
    _clear_cache,
    get_async_openai_client,
    get_cerebras_client,
    get_groq_client,
    get_openai_client,
)
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


class TestOpenAIGPTClientCache:
    """Test client caching functionality for OpenAIGPT."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_openai_client_singleton(self):
        """Test that same config returns same OpenAI client instance."""
        api_key = "test-key-123"
        base_url = "https://api.test.com"

        # Get client twice with same config
        client1 = get_openai_client(api_key=api_key, base_url=base_url)
        client2 = get_openai_client(api_key=api_key, base_url=base_url)

        # Should be same instance
        assert client1 is client2

    def test_openai_client_different_config(self):
        """Test that different configs return different OpenAI client instances."""
        # Different API keys should result in different clients
        client1 = get_openai_client(api_key="key1")
        client2 = get_openai_client(api_key="key2")
        assert client1 is not client2

    def test_async_openai_client_singleton(self):
        """Test that same config returns same AsyncOpenAI client instance."""
        api_key = "test-key-async"

        client1 = get_async_openai_client(api_key=api_key)
        client2 = get_async_openai_client(api_key=api_key)

        assert client1 is client2

    def test_groq_client_singleton(self):
        """Test that same config returns same Groq client instance."""
        api_key = "groq-test-key"

        client1 = get_groq_client(api_key=api_key)
        client2 = get_groq_client(api_key=api_key)

        assert client1 is client2

    def test_mixed_client_types(self):
        """Test that different client types are cached separately."""
        api_key = "same-key-for-all"

        openai_client = get_openai_client(api_key=api_key)
        groq_client = get_groq_client(api_key=api_key)
        cerebras_client = get_cerebras_client(api_key=api_key)

        # All should be different objects despite same API key
        assert openai_client is not groq_client
        assert openai_client is not cerebras_client
        assert groq_client is not cerebras_client

    # Integration tests with OpenAIGPT

    def test_openai_gpt_client_reuse(self):
        """Test that multiple OpenAIGPT instances reuse clients."""
        config = OpenAIGPTConfig(
            api_key="test-key-123",
            chat_model="gpt-4",
        )

        # Create two instances with same config
        gpt1 = OpenAIGPT(config)
        gpt2 = OpenAIGPT(config)

        # They should share the same client instances
        assert gpt1.client is gpt2.client
        assert gpt1.async_client is gpt2.async_client

    def test_openai_gpt_different_config(self):
        """Test that different configs create different clients."""
        config1 = OpenAIGPTConfig(
            api_key="test-key-1",
            chat_model="gpt-4",
        )
        config2 = OpenAIGPTConfig(
            api_key="test-key-2",
            chat_model="gpt-4",
        )

        gpt1 = OpenAIGPT(config1)
        gpt2 = OpenAIGPT(config2)

        # Different API keys should result in different clients
        assert gpt1.client is not gpt2.client
        assert gpt1.async_client is not gpt2.async_client

    def test_use_cached_client_flag(self):
        """Test that use_cached_client config works correctly."""
        # With caching enabled (default)
        config_cached = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            use_cached_client=True,
        )

        gpt1 = OpenAIGPT(config_cached)
        gpt2 = OpenAIGPT(config_cached)
        assert gpt1.client is gpt2.client

        # With caching disabled
        config_no_cache = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            use_cached_client=False,
        )

        gpt3 = OpenAIGPT(config_no_cache)
        gpt4 = OpenAIGPT(config_no_cache)

        # Each instance should have its own client
        assert gpt3.client is not gpt4.client
        assert gpt3.client is not gpt1.client

    @pytest.mark.parametrize("use_cached_client", [True, False])
    def test_concurrent_client_sharing(self, use_cached_client):
        """Test that multiple OpenAIGPT instances share clients correctly."""
        # Create 10 OpenAIGPT instances with same config
        config = OpenAIGPTConfig(
            api_key="test-key-concurrent",
            chat_model="gpt-4",
            use_cached_client=use_cached_client,
        )

        instances = [OpenAIGPT(config) for _ in range(10)]

        if use_cached_client:
            # With caching, they should all share the same sync and async clients
            for i in range(1, 10):
                assert instances[0].client is instances[i].client
                assert instances[0].async_client is instances[i].async_client
        else:
            # Without caching, each should have its own clients
            for i in range(1, 10):
                assert instances[0].client is not instances[i].client
                assert instances[0].async_client is not instances[i].async_client

        # Verify the client is an OpenAI client instance
        assert instances[0].client.__class__.__name__ == "OpenAI"
        assert instances[0].async_client.__class__.__name__ == "AsyncOpenAI"

        # Create instance with different API key - should always get different client
        config_diff = OpenAIGPTConfig(
            api_key="different-test-key",
            chat_model="gpt-4",
            use_cached_client=use_cached_client,
        )
        instance_diff = OpenAIGPT(config_diff)

        # Different API keys should always result in different clients
        assert instance_diff.client is not instances[0].client
        assert instance_diff.async_client is not instances[0].async_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_cached_client", [True, False])
    async def test_concurrent_async_achat(self, use_cached_client):
        """Test that multiple OpenAIGPT instances can make concurrent achat calls."""
        import asyncio

        # Create 10 OpenAIGPT instances with same config
        # API key will be picked up from environment
        config = OpenAIGPTConfig(
            chat_model="gpt-4o-mini",  # Use a cheaper model for testing
            use_cached_client=use_cached_client,
            max_output_tokens=10,  # Keep responses short for testing
        )

        instances = [OpenAIGPT(config) for _ in range(10)]

        # Verify client sharing based on use_cached_client flag
        if use_cached_client:
            # With caching, they should all share the same async client
            for i in range(1, 10):
                assert instances[0].async_client is instances[i].async_client
        else:
            # Without caching, each should have its own client
            for i in range(1, 10):
                assert instances[0].async_client is not instances[i].async_client

        # Define async function to make an achat request
        async def make_achat_request(gpt_instance, idx):
            """Make an async achat request."""
            try:
                response = await gpt_instance.achat(
                    messages=f"what comes after {idx}?",
                    max_tokens=10,
                )
                return idx, "success", response.message
            except Exception as e:
                return idx, "error", f"{type(e).__name__}: {str(e)}"

        # Run all requests concurrently
        tasks = [make_achat_request(inst, i) for i, inst in enumerate(instances)]
        results = await asyncio.gather(*tasks)

        # Verify all requests completed
        assert len(results) == 10

        # Verify they all succeeded (works with or without caching)
        for idx, (req_idx, status, response) in enumerate(results):
            assert req_idx == idx
            assert status == "success"
            # Response should contain the number
            assert str(idx + 1) in response or "zero" in response.lower()

    def test_model_prefix_client_selection(self):
        """Test that different model prefixes activate the correct client types."""
        import os

        # Get the current OPENAI_API_KEY env var value to restore later
        original_openai_key = os.environ.get("OPENAI_API_KEY")

        # Set to dummy value to trigger provider-specific client logic
        if original_openai_key:
            del os.environ["OPENAI_API_KEY"]

        try:
            # Test Groq client
            from langroid.utils.configuration import settings

            original_chat_model = settings.chat_model
            settings.chat_model = ""  # Clear any global override

            groq_config = OpenAIGPTConfig(
                api_key="xxx",  # Use DUMMY_API_KEY value
                chat_model="groq/llama3-8b-8192",
                use_cached_client=True,
            )
            groq_gpt = OpenAIGPT(groq_config)
            assert groq_gpt.client.__class__.__name__ == "Groq"
            assert groq_gpt.async_client.__class__.__name__ == "AsyncGroq"
            assert groq_gpt.is_groq is True
            # Model name should have prefix stripped
            assert groq_gpt.config.chat_model == "llama3-8b-8192"

            # Test standard OpenAI models
            openai_config = OpenAIGPTConfig(
                api_key="test-key",
                chat_model="gpt-4",
                use_cached_client=True,
            )
            openai_gpt = OpenAIGPT(openai_config)
            assert openai_gpt.client.__class__.__name__ == "OpenAI"
            assert openai_gpt.config.chat_model == "gpt-4"

        finally:
            # Restore original settings
            settings.chat_model = original_chat_model
            # Restore original OPENAI_API_KEY
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
