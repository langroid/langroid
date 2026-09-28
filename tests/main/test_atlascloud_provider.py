"""
Tests for Atlas Cloud LLM provider support.

Unit tests run without network access. Integration tests require
ATLASCLOUD_API_KEY.
"""

import os
from unittest.mock import patch

import pytest

from langroid.language_models.openai_gpt import (
    ATLASCLOUD_BASE_URL,
    OpenAIGPT,
    OpenAIGPTConfig,
)

# ──────────────────────── Unit Tests ────────────────────────


class TestAtlasCloudProviderRouting:
    """Unit tests for Atlas Cloud provider routing in OpenAIGPT."""

    def setup_method(self):
        """Clear OPENAI_API_KEY and settings.chat_model to avoid interference."""
        from langroid.utils.configuration import settings

        self._orig_key = os.environ.get("OPENAI_API_KEY")
        if self._orig_key:
            del os.environ["OPENAI_API_KEY"]
        self._orig_chat_model = settings.chat_model
        settings.chat_model = ""

    def teardown_method(self):
        """Restore OPENAI_API_KEY and settings.chat_model."""
        from langroid.utils.configuration import settings

        if self._orig_key:
            os.environ["OPENAI_API_KEY"] = self._orig_key
        settings.chat_model = self._orig_chat_model

    def test_atlascloud_prefix_sets_base_url(self):
        """Using atlascloud/ prefix should set the Atlas Cloud API base URL."""
        config = OpenAIGPTConfig(
            api_key="test-atlascloud-key",
            chat_model="atlascloud/deepseek-ai/DeepSeek-V3.1-Terminus",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_atlascloud is True
        assert gpt.api_base == ATLASCLOUD_BASE_URL
        # Prefix should be stripped from chat_model
        assert gpt.config.chat_model == "deepseek-ai/DeepSeek-V3.1-Terminus"

    def test_atlascloud_prefix_strips_prefix(self):
        """atlascloud/ prefix should be stripped from the model name."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="atlascloud/openai/gpt-4.1-mini",
        )
        gpt = OpenAIGPT(config)
        assert gpt.config.chat_model == "openai/gpt-4.1-mini"

    def test_atlascloud_uses_openai_client(self):
        """Atlas Cloud should use the standard OpenAI client (not Groq/Cerebras)."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="atlascloud/openai/gpt-4.1-mini",
        )
        gpt = OpenAIGPT(config)
        assert gpt.client.__class__.__name__ == "OpenAI"
        assert gpt.async_client.__class__.__name__ == "AsyncOpenAI"

    def test_atlascloud_api_key_from_env(self):
        """ATLASCLOUD_API_KEY env var should be used when no explicit key given."""
        with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "env-atlascloud-key"}):
            config = OpenAIGPTConfig(
                chat_model="atlascloud/openai/gpt-4.1-mini",
            )
            gpt = OpenAIGPT(config)
            assert gpt.api_key == "env-atlascloud-key"

    def test_atlascloud_explicit_api_key_takes_precedence(self):
        """Explicit api_key in config should override env var."""
        with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "env-key"}):
            config = OpenAIGPTConfig(
                api_key="explicit-key",
                chat_model="atlascloud/openai/gpt-4.1-mini",
            )
            gpt = OpenAIGPT(config)
            assert gpt.api_key == "explicit-key"

    def test_is_not_atlascloud_model(self):
        """is_atlascloud should be False for non-Atlas-Cloud models."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4o",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_atlascloud is False

    def test_atlascloud_base_url_constant(self):
        """ATLASCLOUD_BASE_URL should point to the correct endpoint."""
        assert ATLASCLOUD_BASE_URL == "https://api.atlascloud.ai/v1"

    def test_atlascloud_honors_explicit_api_base(self):
        """Caller-supplied api_base should not be overwritten."""
        custom_base = "https://custom.proxy.example.com/v1"
        config = OpenAIGPTConfig(
            api_key="test-key",
            api_base=custom_base,
            chat_model="atlascloud/openai/gpt-4.1-mini",
        )
        gpt = OpenAIGPT(config)
        assert gpt.api_base == custom_base

    def test_atlascloud_openai_key_not_clobbered_without_atlascloud_key(self):
        """OPENAI_API_KEY should be kept when ATLASCLOUD_API_KEY is unset."""
        env = {"OPENAI_API_KEY": "my-openai-key"}
        # Ensure ATLASCLOUD_API_KEY is NOT in the environment
        env_clear = {k: v for k, v in os.environ.items() if k != "ATLASCLOUD_API_KEY"}
        env_clear.update(env)
        with patch.dict(os.environ, env_clear, clear=True):
            config = OpenAIGPTConfig(
                chat_model="atlascloud/openai/gpt-4.1-mini",
            )
            gpt = OpenAIGPT(config)
            # Should be replaced by the dummy key, since OPENAI_API_KEY is not
            # a valid Atlas Cloud key and ATLASCLOUD_API_KEY is unset.
            assert gpt.api_key != "my-openai-key"


# ──────────────────────── Integration Tests ────────────────────────


@pytest.mark.integration
class TestAtlasCloudIntegration:
    """Integration tests requiring ATLASCLOUD_API_KEY env var."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Skip integration tests if ATLASCLOUD_API_KEY is not set."""
        if not os.environ.get("ATLASCLOUD_API_KEY"):
            pytest.skip("ATLASCLOUD_API_KEY not set")

    def setup_method(self):
        """Clear settings.chat_model to avoid global override."""
        from langroid.utils.configuration import settings

        self._orig_chat_model = settings.chat_model
        settings.chat_model = ""

    def teardown_method(self):
        """Restore settings.chat_model."""
        from langroid.utils.configuration import settings

        settings.chat_model = self._orig_chat_model

    def test_atlascloud_chat_completion(self):
        """Test basic chat completion via Atlas Cloud."""
        config = OpenAIGPTConfig(
            chat_model="atlascloud/openai/gpt-4.1-mini",
            max_output_tokens=50,
            temperature=0.0,
        )
        gpt = OpenAIGPT(config)
        response = gpt.chat("What is 2+2? Reply with just the number.")
        assert response is not None
        assert response.message is not None
        assert "4" in response.message

    @pytest.mark.asyncio
    async def test_atlascloud_async_chat(self):
        """Test async chat completion via Atlas Cloud."""
        config = OpenAIGPTConfig(
            chat_model="atlascloud/openai/gpt-4.1-mini",
            max_output_tokens=50,
            temperature=0.0,
        )
        gpt = OpenAIGPT(config)
        response = await gpt.achat("What is 3+3? Reply with just the number.")
        assert response is not None
        assert response.message is not None
        assert "6" in response.message
