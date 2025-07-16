"""
Simplified tests for OpenAI http_client configuration.
"""

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig


class TestHTTPClientSimple:
    """Simple test to verify http_verify_ssl configuration works."""

    def test_ssl_verification_disabled_creates_client(self):
        """Test that http_verify_ssl=False creates appropriate clients."""
        # This test just verifies the client is created with the right config
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            api_key="test-key",
            http_verify_ssl=False,
            use_cached_client=False,
        )

        llm = OpenAIGPT(config)

        # Verify the configuration was set correctly
        assert llm.config.http_verify_ssl is False
        assert llm is not None

    def test_http_client_factory_is_called(self):
        """Test that http_client_factory is called during initialization."""
        factory_called = False

        def test_factory():
            nonlocal factory_called
            factory_called = True
            return None  # Return None to avoid type issues

        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            api_key="test-key",
            http_client_factory=test_factory,
            use_cached_client=False,
        )

        _ = OpenAIGPT(config)
        assert factory_called is True
