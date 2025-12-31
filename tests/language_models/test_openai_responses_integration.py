import os

import pytest

from langroid.language_models.base import LanguageModel, LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
class TestIntegration:
    def test_create_factory(self):
        """LanguageModel.create properly routes to OpenAIResponses."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
        )

        llm = LanguageModel.create(config)

        assert isinstance(llm, OpenAIResponses)
        assert llm.config.type == "openai_responses"

    def test_import_availability(self):
        """OpenAIResponses is available from package imports."""
        from langroid.language_models import OpenAIResponses, OpenAIResponsesConfig

        assert OpenAIResponses is not None
        assert OpenAIResponsesConfig is not None

    def test_config_inheritance(self):
        """OpenAIResponsesConfig inherits from OpenAIGPTConfig."""
        from langroid.language_models.openai_gpt import OpenAIGPTConfig

        config = OpenAIResponsesConfig(
            chat_model="gpt-4o-mini",
            temperature=0.5,
            max_output_tokens=100,
        )

        # Should inherit all OpenAIGPTConfig fields
        assert isinstance(config, OpenAIGPTConfig)
        assert config.temperature == 0.5
        assert config.max_output_tokens == 100
        assert config.type == "openai_responses"

    def test_basic_functionality(self):
        """Basic chat functionality works through factory creation."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )

        llm = LanguageModel.create(config)

        messages = [
            LLMMessage(role=Role.USER, content="Say 'integration test'"),
        ]

        response = llm.chat(messages, max_tokens=20)

        assert response.message is not None
        assert (
            "integration" in response.message.lower()
            or "test" in response.message.lower()
        )
        assert response.usage is not None

    def test_all_methods_present(self):
        """All required LanguageModel methods are implemented."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
        llm = OpenAIResponses(config)

        # Check required methods exist
        assert hasattr(llm, "chat")
        assert hasattr(llm, "achat")
        assert hasattr(llm, "generate")
        assert hasattr(llm, "agenerate")
        assert hasattr(llm, "set_stream")
        assert hasattr(llm, "get_stream")
        assert hasattr(llm, "reset_usage_cost")
        assert hasattr(llm, "update_usage_cost")

    def test_stream_toggle(self):
        """Stream setting can be toggled."""
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o-mini",
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Initially False
        assert llm.get_stream() is False

        # Toggle to True
        prev = llm.set_stream(True)
        assert prev is False
        assert llm.get_stream() is True

        # Toggle back to False
        prev = llm.set_stream(False)
        assert prev is True
        assert llm.get_stream() is False

    def test_model_info_available(self):
        """Model info is available for cost calculations."""
        from langroid.language_models.model_info import get_model_info

        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")

        # Should be able to get model info
        info = get_model_info(config.chat_model)
        assert info is not None
        assert info.input_cost_per_million > 0
        assert info.output_cost_per_million > 0

    def test_environment_variable_config(self):
        """Config can use environment variables."""
        # OpenAIResponsesConfig should inherit env var support from OpenAIGPTConfig
        # Set env var (if not already set)
        import os

        original_key = os.environ.get("OPENAI_API_KEY")

        if not original_key:
            os.environ["OPENAI_API_KEY"] = "test-key-from-env"

        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")

        # Should pick up API key from env
        assert config.api_key is not None

        # Restore original
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
