"""Tests for OpenAI Responses API routing mechanism."""

from unittest.mock import MagicMock, patch

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LanguageModel
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


class TestRoutingMechanism:
    """Test the routing from OpenAIGPT to OpenAIResponses based on flag."""

    def test_routing_to_responses_api(self):
        """Test that use_responses_api=True routes to OpenAIResponses."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4o", use_responses_api=True, api_key="test-key"
        )

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = LanguageModel.create(config)

            # Should create OpenAIResponses instance
            assert isinstance(llm, OpenAIResponses)
            assert not isinstance(llm, OpenAIGPT)

    def test_routing_to_chat_completions(self):
        """Test that use_responses_api=False routes to OpenAIGPT."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4o", use_responses_api=False, api_key="test-key"
        )

        with patch("langroid.language_models.openai_gpt.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = LanguageModel.create(config)

            # Should create OpenAIGPT instance
            assert isinstance(llm, OpenAIGPT)
            assert not isinstance(llm, OpenAIResponses)

    def test_default_routing(self):
        """Test that default (no flag) routes to OpenAIGPT."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4o",
            api_key="test-key",
            # use_responses_api not specified, should default to False
        )

        with patch("langroid.language_models.openai_gpt.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = LanguageModel.create(config)

            # Should create OpenAIGPT instance by default
            assert isinstance(llm, OpenAIGPT)
            assert not isinstance(llm, OpenAIResponses)

    def test_explicit_responses_config(self):
        """Test that OpenAIResponsesConfig always routes to OpenAIResponses."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o", api_key="test-key")

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = LanguageModel.create(config)

            # Should create OpenAIResponses instance
            assert isinstance(llm, OpenAIResponses)
            assert not isinstance(llm, OpenAIGPT)

    def test_openai_responses_accepts_gpt_config(self):
        """Test that OpenAIResponses can be initialized with OpenAIGPTConfig."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4o",
            use_responses_api=True,
            api_key="test-key",
            temperature=0.7,
            max_output_tokens=1000,
        )

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Should be able to create OpenAIResponses with OpenAIGPTConfig
            llm = OpenAIResponses(config)

            assert isinstance(llm.config, OpenAIResponsesConfig)
            assert llm.config.chat_model == "gpt-4o"
            assert llm.config.temperature == 0.7
            assert llm.config.max_output_tokens == 1000


class TestChatAgentCompatibility:
    """Test that ChatAgent works with both implementations."""

    def test_chat_agent_with_responses_api(self):
        """Test that ChatAgent works with OpenAIResponses via routing."""
        config = ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="gpt-4o", use_responses_api=True, api_key="test-key"
            )
        )

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            agent = ChatAgent(config)

            # Should have OpenAIResponses as LLM
            assert isinstance(agent.llm, OpenAIResponses)

            # Check that methods recognize OpenAIResponses
            agent.llm.supports_strict_tools = True
            agent.llm.supports_json_schema = True
            agent.llm.config.parallel_tool_calls = False

            # These should return True now that OpenAIResponses is recognized
            assert agent._strict_tools_available() is True
            assert agent._json_schema_available() is True

    def test_chat_agent_with_chat_completions(self):
        """Test that ChatAgent still works with OpenAIGPT."""
        config = ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="gpt-4o", use_responses_api=False, api_key="test-key"
            )
        )

        with patch("langroid.language_models.openai_gpt.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            agent = ChatAgent(config)

            # Should have OpenAIGPT as LLM
            assert isinstance(agent.llm, OpenAIGPT)

            # Check that methods still work with OpenAIGPT
            agent.llm.supports_strict_tools = True
            agent.llm.supports_json_schema = True
            agent.llm.config.parallel_tool_calls = False

            # These should return True for OpenAIGPT
            assert agent._strict_tools_available() is True
            assert agent._json_schema_available() is True


class TestPropertiesSupport:
    """Test that OpenAIResponses has required properties."""

    def test_supports_strict_tools_property(self):
        """Test that OpenAIResponses has supports_strict_tools property."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o", api_key="test-key")

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = OpenAIResponses(config)

            # Should have the property and return True for gpt-4 models
            assert hasattr(llm, "supports_strict_tools")
            assert llm.supports_strict_tools is True

    def test_supports_json_schema_property(self):
        """Test that OpenAIResponses has supports_json_schema property."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o", api_key="test-key")

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = OpenAIResponses(config)

            # Should have the property and return True for gpt-4 models
            assert hasattr(llm, "supports_json_schema")
            assert llm.supports_json_schema is True

    def test_properties_for_o1_models(self):
        """Test that properties work for o1 reasoning models."""
        config = OpenAIResponsesConfig(chat_model="o1-preview", api_key="test-key")

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = OpenAIResponses(config)

            # Should return True for o1 models
            assert llm.supports_strict_tools is True
            assert llm.supports_json_schema is True

    def test_properties_for_legacy_models(self):
        """Test that properties work for gpt-3.5 models."""
        config = OpenAIResponsesConfig(chat_model="gpt-3.5-turbo", api_key="test-key")

        with patch("langroid.language_models.openai_responses.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            llm = OpenAIResponses(config)

            # Should return True for gpt-3.5 models
            assert llm.supports_strict_tools is True
            assert llm.supports_json_schema is True
