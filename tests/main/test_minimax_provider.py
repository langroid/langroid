"""
Tests for MiniMax LLM provider support.

Unit tests run without network access. Integration tests require MINIMAX_API_KEY.
"""

import os
from unittest.mock import patch

import pytest

from langroid.language_models.model_info import (
    MODEL_INFO,
    MiniMaxModel,
    ModelInfo,
    ModelProvider,
    get_model_info,
)
from langroid.language_models.openai_gpt import (
    MINIMAX_BASE_URL,
    OpenAIGPT,
    OpenAIGPTConfig,
)

# ──────────────────────── Unit Tests ────────────────────────


class TestMiniMaxModelInfo:
    """Unit tests for MiniMax model info registry."""

    def test_minimax_provider_enum(self):
        """MINIMAX is a valid ModelProvider enum member."""
        assert ModelProvider.MINIMAX == "minimax"
        assert ModelProvider.MINIMAX.value == "minimax"

    def test_minimax_model_enum_values(self):
        """MiniMaxModel enum contains the expected model names."""
        assert MiniMaxModel.MINIMAX_M2_7.value == "MiniMax-M2.7"
        assert MiniMaxModel.MINIMAX_M2_7_HIGHSPEED.value == "MiniMax-M2.7-highspeed"
        assert MiniMaxModel.MINIMAX_M2_5.value == "MiniMax-M2.5"
        assert MiniMaxModel.MINIMAX_M2_5_HIGHSPEED.value == "MiniMax-M2.5-highspeed"
        assert MiniMaxModel.MINIMAX_M2_1.value == "MiniMax-M2.1"
        assert MiniMaxModel.MINIMAX_M2_1_HIGHSPEED.value == "MiniMax-M2.1-highspeed"
        assert MiniMaxModel.MINIMAX_M2.value == "MiniMax-M2"

    @pytest.mark.parametrize(
        "model",
        [
            MiniMaxModel.MINIMAX_M2_7,
            MiniMaxModel.MINIMAX_M2_7_HIGHSPEED,
            MiniMaxModel.MINIMAX_M2_5,
            MiniMaxModel.MINIMAX_M2_5_HIGHSPEED,
            MiniMaxModel.MINIMAX_M2_1,
            MiniMaxModel.MINIMAX_M2_1_HIGHSPEED,
            MiniMaxModel.MINIMAX_M2,
        ],
    )
    def test_model_info_registered(self, model):
        """Each MiniMax model has a ModelInfo entry in the registry."""
        assert model.value in MODEL_INFO
        info = MODEL_INFO[model.value]
        assert isinstance(info, ModelInfo)
        assert info.provider == ModelProvider.MINIMAX
        assert info.context_length > 0
        assert info.max_output_tokens > 0

    def test_get_model_info_minimax(self):
        """get_model_info returns correct info for MiniMax models."""
        info = get_model_info("MiniMax-M2.7")
        assert info.provider == ModelProvider.MINIMAX
        assert info.context_length == 204_800
        assert info.has_structured_output is True

    def test_model_info_m27_context(self):
        """M2.7 models have 204K context length."""
        info_m27 = MODEL_INFO[MiniMaxModel.MINIMAX_M2_7.value]
        info_m27hs = MODEL_INFO[MiniMaxModel.MINIMAX_M2_7_HIGHSPEED.value]
        assert info_m27.context_length == 204_800
        assert info_m27hs.context_length == 204_800

    def test_model_info_m25_context(self):
        """M2.5 models have 196K context length."""
        info_m25 = MODEL_INFO[MiniMaxModel.MINIMAX_M2_5.value]
        info_m25hs = MODEL_INFO[MiniMaxModel.MINIMAX_M2_5_HIGHSPEED.value]
        assert info_m25.context_length == 196_608
        assert info_m25hs.context_length == 196_608

    def test_highspeed_models_cheaper(self):
        """Highspeed variants should cost less than standard variants."""
        m27 = MODEL_INFO[MiniMaxModel.MINIMAX_M2_7.value]
        m27hs = MODEL_INFO[MiniMaxModel.MINIMAX_M2_7_HIGHSPEED.value]
        assert m27hs.input_cost_per_million < m27.input_cost_per_million
        assert m27hs.output_cost_per_million < m27.output_cost_per_million


class TestMiniMaxProviderRouting:
    """Unit tests for MiniMax provider routing in OpenAIGPT."""

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

    def test_minimax_prefix_sets_base_url(self):
        """Using minimax/ prefix should set the MiniMax API base URL."""
        config = OpenAIGPTConfig(
            api_key="test-minimax-key",
            chat_model="minimax/MiniMax-M2.7",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_minimax is True
        assert gpt.api_base == MINIMAX_BASE_URL
        # Prefix should be stripped from chat_model
        assert gpt.config.chat_model == "MiniMax-M2.7"

    def test_minimax_prefix_strips_prefix(self):
        """minimax/ prefix should be stripped from the model name."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.7-highspeed",
        )
        gpt = OpenAIGPT(config)
        assert gpt.config.chat_model == "MiniMax-M2.7-highspeed"

    def test_minimax_uses_openai_client(self):
        """MiniMax should use the standard OpenAI client (not Groq/Cerebras)."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.7",
        )
        gpt = OpenAIGPT(config)
        assert gpt.client.__class__.__name__ == "OpenAI"
        assert gpt.async_client.__class__.__name__ == "AsyncOpenAI"

    def test_minimax_api_key_from_env(self):
        """MINIMAX_API_KEY env var should be used when no explicit key is given."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}):
            config = OpenAIGPTConfig(
                chat_model="minimax/MiniMax-M2.7",
            )
            gpt = OpenAIGPT(config)
            assert gpt.api_key == "env-minimax-key"

    def test_minimax_explicit_api_key_takes_precedence(self):
        """Explicit api_key in config should override env var."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            config = OpenAIGPTConfig(
                api_key="explicit-key",
                chat_model="minimax/MiniMax-M2.7",
            )
            gpt = OpenAIGPT(config)
            assert gpt.api_key == "explicit-key"

    def test_is_minimax_model_with_prefix(self):
        """is_minimax_model() should return True for minimax/ prefixed models."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.5",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_minimax_model() is True

    def test_is_minimax_model_with_enum(self):
        """is_minimax_model() should return True for MiniMaxModel enum values."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.7",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_minimax_model() is True

    def test_is_not_minimax_model(self):
        """is_minimax_model() should return False for non-MiniMax models."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="gpt-4o",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_minimax is False

    def test_minimax_m25_highspeed_routing(self):
        """M2.5-highspeed model should route correctly."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.5-highspeed",
        )
        gpt = OpenAIGPT(config)
        assert gpt.is_minimax is True
        assert gpt.config.chat_model == "MiniMax-M2.5-highspeed"
        assert gpt.api_base == MINIMAX_BASE_URL

    def test_minimax_base_url_constant(self):
        """MINIMAX_BASE_URL should point to the correct endpoint."""
        assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"

    def test_minimax_honors_explicit_api_base(self):
        """Caller-supplied api_base should not be overwritten."""
        custom_base = "https://api.minimaxi.com/v1"
        config = OpenAIGPTConfig(
            api_key="test-key",
            api_base=custom_base,
            chat_model="minimax/MiniMax-M2.5",
        )
        gpt = OpenAIGPT(config)
        assert gpt.api_base == custom_base

    def test_minimax_supports_json_schema(self):
        """MiniMax models with has_structured_output should have JSON schema support.

        Regression test: supports_json_schema was computed before the minimax/
        prefix was stripped, so self.info() looked up 'minimax/MiniMax-M2.7'
        (not in MODEL_INFO) and incorrectly disabled JSON schema support.
        """
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.7",
        )
        gpt = OpenAIGPT(config)
        # MiniMax-M2.7 has has_structured_output=True in MODEL_INFO
        assert gpt.supports_json_schema is True

    def test_minimax_supports_strict_tools(self):
        """MiniMax models should support strict tools (OpenAI-compatible API)."""
        config = OpenAIGPTConfig(
            api_key="test-key",
            chat_model="minimax/MiniMax-M2.7",
        )
        gpt = OpenAIGPT(config)
        assert gpt.supports_strict_tools is True

    def test_minimax_openai_key_not_clobbered_without_minimax_key(self):
        """OPENAI_API_KEY should be kept when MINIMAX_API_KEY is unset."""
        env = {"OPENAI_API_KEY": "my-openai-key"}
        # Ensure MINIMAX_API_KEY is NOT in the environment
        env_clear = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        env_clear.update(env)
        with patch.dict(os.environ, env_clear, clear=True):
            config = OpenAIGPTConfig(
                chat_model="minimax/MiniMax-M2.5",
            )
            gpt = OpenAIGPT(config)
            # Should keep the OPENAI_API_KEY value, not replace with dummy
            assert gpt.api_key == "my-openai-key"


class TestMiniMaxExports:
    """Unit tests for MiniMax exports in __init__.py."""

    def test_minimax_model_importable(self):
        """MiniMaxModel should be importable from langroid.language_models."""
        from langroid.language_models import MiniMaxModel

        assert MiniMaxModel.MINIMAX_M2_7.value == "MiniMax-M2.7"


# ──────────────────────── Integration Tests ────────────────────────


@pytest.mark.integration
class TestMiniMaxIntegration:
    """Integration tests requiring MINIMAX_API_KEY env var."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Skip integration tests if MINIMAX_API_KEY is not set."""
        if not os.environ.get("MINIMAX_API_KEY"):
            pytest.skip("MINIMAX_API_KEY not set")

    def setup_method(self):
        """Clear settings.chat_model to avoid global override."""
        from langroid.utils.configuration import settings

        self._orig_chat_model = settings.chat_model
        settings.chat_model = ""

    def teardown_method(self):
        """Restore settings.chat_model."""
        from langroid.utils.configuration import settings

        settings.chat_model = self._orig_chat_model

    def test_minimax_chat_completion(self):
        """Test basic chat completion with MiniMax M2.7."""
        config = OpenAIGPTConfig(
            chat_model="minimax/MiniMax-M2.7",
            max_output_tokens=50,
            temperature=0.0,
        )
        gpt = OpenAIGPT(config)
        response = gpt.chat("What is 2+2? Reply with just the number.")
        assert response is not None
        assert response.message is not None
        assert "4" in response.message

    def test_minimax_highspeed_chat(self):
        """Test chat completion with MiniMax M2.7-highspeed."""
        config = OpenAIGPTConfig(
            chat_model="minimax/MiniMax-M2.7-highspeed",
            max_output_tokens=50,
            temperature=0.0,
        )
        gpt = OpenAIGPT(config)
        response = gpt.chat("Say hello in exactly one word.")
        assert response is not None
        assert response.message is not None
        assert len(response.message.strip()) > 0

    @pytest.mark.asyncio
    async def test_minimax_async_chat(self):
        """Test async chat completion with MiniMax."""
        config = OpenAIGPTConfig(
            chat_model="minimax/MiniMax-M2.7-highspeed",
            max_output_tokens=50,
            temperature=0.0,
        )
        gpt = OpenAIGPT(config)
        response = await gpt.achat("What is 3+3? Reply with just the number.")
        assert response is not None
        assert response.message is not None
        assert "6" in response.message
