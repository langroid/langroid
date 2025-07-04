"""
Tests for OpenAIGPT client caching functionality.
"""

from httpx import Timeout

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
        # Different API keys
        client1 = get_openai_client(api_key="key1")
        client2 = get_openai_client(api_key="key2")
        assert client1 is not client2

        # Different base URLs
        client3 = get_openai_client(api_key="key1", base_url="https://api1.com")
        client4 = get_openai_client(api_key="key1", base_url="https://api2.com")
        assert client3 is not client4

        # Different organizations
        client5 = get_openai_client(api_key="key1", organization="org1")
        client6 = get_openai_client(api_key="key1", organization="org2")
        assert client5 is not client6

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

    def test_groq_client_different_keys(self):
        """Test that different API keys return different Groq clients."""
        client1 = get_groq_client(api_key="groq-key1")
        client2 = get_groq_client(api_key="groq-key2")

        assert client1 is not client2

    def test_cerebras_client_singleton(self):
        """Test that same config returns same Cerebras client instance."""
        api_key = "cerebras-test-key"

        client1 = get_cerebras_client(api_key=api_key)
        client2 = get_cerebras_client(api_key=api_key)

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

    def test_timeout_handling(self):
        """Test that timeout values are properly handled in cache key."""
        api_key = "test-key"

        # Same timeout value and type
        client1 = get_openai_client(api_key=api_key, timeout=30.0)
        client2 = get_openai_client(api_key=api_key, timeout=30.0)
        assert client1 is client2

        # Note: 30 and 30.0 are treated as different due to string representation
        # This is intentional to avoid any potential issues with type differences
        client_int = get_openai_client(api_key=api_key, timeout=30)
        client_float = get_openai_client(api_key=api_key, timeout=30.0)
        assert client_int is not client_float

        # Different timeout values
        client3 = get_openai_client(api_key=api_key, timeout=60.0)
        assert client1 is not client3

        # Timeout object
        timeout_obj = Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0)
        client4 = get_openai_client(api_key=api_key, timeout=timeout_obj)
        client5 = get_openai_client(api_key=api_key, timeout=timeout_obj)
        assert client4 is client5

    def test_headers_handling(self):
        """Test that headers are properly handled in cache key."""
        api_key = "test-key"

        # Same headers
        headers1 = {"X-Custom": "value1", "X-Other": "value2"}
        client1 = get_openai_client(api_key=api_key, default_headers=headers1)
        client2 = get_openai_client(api_key=api_key, default_headers=headers1)
        assert client1 is client2

        # Different headers
        headers2 = {"X-Custom": "value3"}
        client3 = get_openai_client(api_key=api_key, default_headers=headers2)
        assert client1 is not client3

        # No headers vs headers
        client4 = get_openai_client(api_key=api_key)
        assert client1 is not client4

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

    def test_groq_client_reuse(self):
        """Test that Groq clients are reused."""
        config = OpenAIGPTConfig(
            api_key="groq-test-key",
            chat_model="groq/llama3-8b-8192",
        )

        gpt1 = OpenAIGPT(config)
        gpt2 = OpenAIGPT(config)

        assert gpt1.client is gpt2.client
        assert gpt1.async_client is gpt2.async_client

    def test_cerebras_client_reuse(self):
        """Test that Cerebras clients are reused."""
        config = OpenAIGPTConfig(
            api_key="cerebras-test-key",
            chat_model="cerebras/llama3-8b",
        )

        gpt1 = OpenAIGPT(config)
        gpt2 = OpenAIGPT(config)

        assert gpt1.client is gpt2.client
        assert gpt1.async_client is gpt2.async_client

    def test_base_url_difference(self):
        """Test that different base URLs create different clients."""
        config1 = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            api_base="https://api1.openai.com",
        )
        config2 = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            api_base="https://api2.openai.com",
        )

        gpt1 = OpenAIGPT(config1)
        gpt2 = OpenAIGPT(config2)

        assert gpt1.client is not gpt2.client
        assert gpt1.async_client is not gpt2.async_client

    def test_headers_difference(self):
        """Test that different headers create different clients."""
        config1 = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            headers={"X-Custom": "value1"},
        )
        config2 = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4",
            headers={"X-Custom": "value2"},
        )

        gpt1 = OpenAIGPT(config1)
        gpt2 = OpenAIGPT(config2)

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

    def test_concurrent_client_sharing(self):
        """Test that multiple OpenAIGPT instances share clients correctly."""
        # Create 10 OpenAIGPT instances with same config
        config = OpenAIGPTConfig(
            api_key="test-key-concurrent",
            chat_model="gpt-4",
            use_cached_client=True,
        )

        instances = [OpenAIGPT(config) for _ in range(10)]

        # Verify they all share the same sync and async clients
        for i in range(1, 10):
            assert instances[0].client is instances[i].client
            assert instances[0].async_client is instances[i].async_client

        # Verify the client is an OpenAI client instance
        assert instances[0].client.__class__.__name__ == "OpenAI"
        assert instances[0].async_client.__class__.__name__ == "AsyncOpenAI"

        # Create instance with different API key - should get different client
        config_diff = OpenAIGPTConfig(
            api_key="different-test-key",
            chat_model="gpt-4",
            use_cached_client=True,
        )
        instance_diff = OpenAIGPT(config_diff)

        assert instance_diff.client is not instances[0].client
        assert instance_diff.async_client is not instances[0].async_client

    def test_client_sharing_different_models(self):
        """Test client sharing with different model configurations."""
        # Create instances with different models but same API key
        configs = [
            OpenAIGPTConfig(
                api_key="test-key-models",
                chat_model="gpt-4",
                use_cached_client=True,
            ),
            OpenAIGPTConfig(
                api_key="test-key-models",
                chat_model="gpt-4o",
                use_cached_client=True,
            ),
            OpenAIGPTConfig(
                api_key="test-key-models",
                chat_model="gpt-3.5-turbo",
                use_cached_client=True,
            ),
        ]

        # Create 3 instances of each config (9 total)
        instances = []
        for config in configs:
            instances.extend([OpenAIGPT(config) for _ in range(3)])

        # Verify client sharing within same config
        assert instances[0].client is instances[1].client  # gpt-4
        assert instances[0].async_client is instances[1].async_client
        assert instances[3].client is instances[4].client  # gpt-4o
        assert instances[3].async_client is instances[4].async_client
        assert instances[6].client is instances[7].client  # gpt-3.5
        assert instances[6].async_client is instances[7].async_client

        # Verify all configs share the same clients
        # (same API key and base URL means same client, regardless of model)
        assert instances[0].client is instances[3].client
        assert instances[0].client is instances[6].client
        assert instances[3].client is instances[6].client

        assert instances[0].async_client is instances[3].async_client
        assert instances[0].async_client is instances[6].async_client
        assert instances[3].async_client is instances[6].async_client

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

            # Test Cerebras client
            cerebras_config = OpenAIGPTConfig(
                api_key="xxx",  # Use DUMMY_API_KEY value
                chat_model="cerebras/llama3-8b",
                use_cached_client=True,
            )
            cerebras_gpt = OpenAIGPT(cerebras_config)
            assert cerebras_gpt.client.__class__.__name__ == "Cerebras"
            assert cerebras_gpt.async_client.__class__.__name__ == "AsyncCerebras"
            assert cerebras_gpt.is_cerebras is True
            assert cerebras_gpt.config.chat_model == "llama3-8b"

            # Test OpenAI client for various prefixes that use OpenAI client
            prefixes_using_openai = [
                ("gemini/gemini-pro", "gemini-pro", "is_gemini"),
                ("deepseek/deepseek-coder", "deepseek-coder", "is_deepseek"),
                (
                    "glhf/hf:meta-llama/Llama-3.1-70B-Instruct",
                    "hf:meta-llama/Llama-3.1-70B-Instruct",
                    "is_glhf",
                ),
                (
                    "openrouter/anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3.5-sonnet",
                    "is_openrouter",
                ),
                ("litellm-proxy/gpt-4", "gpt-4", "is_litellm_proxy"),
            ]

            for model_name, expected_stripped, flag_name in prefixes_using_openai:
                config = OpenAIGPTConfig(
                    api_key="xxx",  # Use DUMMY_API_KEY value
                    chat_model=model_name,
                    use_cached_client=True,
                )
                gpt = OpenAIGPT(config)
                assert gpt.client.__class__.__name__ == "OpenAI"
                assert gpt.async_client.__class__.__name__ == "AsyncOpenAI"
                assert getattr(gpt, flag_name) is True
                assert gpt.config.chat_model == expected_stripped

            # Test special cases for local models
            local_config = OpenAIGPTConfig(
                api_key="xxx",
                chat_model="local/localhost:8000/v1",
                use_cached_client=True,
            )
            local_gpt = OpenAIGPT(local_config)
            assert local_gpt.client.__class__.__name__ == "OpenAI"
            assert local_gpt.api_base == "http://localhost:8000/v1"

            # Test ollama
            ollama_config = OpenAIGPTConfig(
                api_key="xxx",
                chat_model="ollama/llama2",
                use_cached_client=True,
            )
            ollama_gpt = OpenAIGPT(ollama_config)
            assert ollama_gpt.client.__class__.__name__ == "OpenAI"
            assert ollama_gpt.config.chat_model == "llama2"
            # API key is only changed to "ollama" if it matches OPENAI_API_KEY from env
            # Since we unset the env var, it will remain as configured
            assert ollama_gpt.api_key == "xxx"

            # Test standard OpenAI models
            openai_config = OpenAIGPTConfig(
                api_key="test-key",
                chat_model="gpt-4",
                use_cached_client=True,
            )
            openai_gpt = OpenAIGPT(openai_config)
            assert openai_gpt.client.__class__.__name__ == "OpenAI"
            assert openai_gpt.config.chat_model == "gpt-4"
            # api_base may be set from config or env, just verify it's not a
            # provider-specific URL
            if openai_gpt.api_base:
                assert "gemini" not in openai_gpt.api_base
                assert "deepseek" not in openai_gpt.api_base

        finally:
            # Restore original settings
            settings.chat_model = original_chat_model
            # Restore original OPENAI_API_KEY
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key

    def test_client_caching_across_providers(self):
        """Test that client caching works correctly across different providers."""
        import os

        from langroid.utils.configuration import settings

        # Save original values
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        original_groq_key = os.environ.get("GROQ_API_KEY")
        original_cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        original_chat_model = settings.chat_model

        # Clear to ensure provider-specific clients are used
        if original_openai_key:
            del os.environ["OPENAI_API_KEY"]
        if original_groq_key:
            del os.environ["GROQ_API_KEY"]
        if original_cerebras_key:
            del os.environ["CEREBRAS_API_KEY"]
        settings.chat_model = ""

        try:
            # Create multiple Groq instances - should share client
            groq_configs = [
                OpenAIGPTConfig(
                    api_key="xxx",
                    chat_model="groq/llama3-8b-8192",
                    use_cached_client=True,
                ),
                OpenAIGPTConfig(
                    api_key="xxx",
                    chat_model="groq/mixtral-8x7b-32768",
                    use_cached_client=True,
                ),
            ]
            groq_instances = [OpenAIGPT(cfg) for cfg in groq_configs]

            # Both should share the same Groq client (same API key)
            assert groq_instances[0].client is groq_instances[1].client
            assert groq_instances[0].client.__class__.__name__ == "Groq"

            # Create Cerebras instances with different keys - should NOT share
            cerebras_configs = [
                OpenAIGPTConfig(
                    api_key="cerebras-key-1",
                    chat_model="cerebras/llama3-8b",
                    use_cached_client=True,
                ),
                OpenAIGPTConfig(
                    api_key="cerebras-key-2",
                    chat_model="cerebras/llama3-8b",
                    use_cached_client=True,
                ),
            ]
            cerebras_instances = [OpenAIGPT(cfg) for cfg in cerebras_configs]

            # Different API keys should result in different clients
            assert cerebras_instances[0].client is not cerebras_instances[1].client
            assert cerebras_instances[0].client.__class__.__name__ == "Cerebras"

            # Test that OpenAI-based providers with different base URLs get
            # different clients
            gemini_config = OpenAIGPTConfig(
                api_key="xxx",
                chat_model="gemini/gemini-pro",
                use_cached_client=True,
            )
            deepseek_config = OpenAIGPTConfig(
                api_key="xxx",
                chat_model="deepseek/deepseek-coder",
                use_cached_client=True,
            )

            gemini_gpt = OpenAIGPT(gemini_config)
            deepseek_gpt = OpenAIGPT(deepseek_config)

            # Different base URLs should result in different OpenAI clients
            assert gemini_gpt.client is not deepseek_gpt.client
            assert gemini_gpt.client.__class__.__name__ == "OpenAI"
            assert deepseek_gpt.client.__class__.__name__ == "OpenAI"
            assert gemini_gpt.api_base != deepseek_gpt.api_base

        finally:
            # Restore original values
            settings.chat_model = original_chat_model
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            if original_groq_key:
                os.environ["GROQ_API_KEY"] = original_groq_key
            if original_cerebras_key:
                os.environ["CEREBRAS_API_KEY"] = original_cerebras_key
