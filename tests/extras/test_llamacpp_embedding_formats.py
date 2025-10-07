"""
Unit tests for LlamaCppServerEmbeddings response format handling.
Tests the _extract_embedding method with various llama.cpp response formats.
"""

from unittest.mock import Mock, patch

import pytest

from langroid.embedding_models.models import (
    LlamaCppServerEmbeddings,
    LlamaCppServerEmbeddingsConfig,
)


@pytest.fixture
def llamacpp_model():
    """Create a LlamaCppServerEmbeddings instance for testing"""
    config = LlamaCppServerEmbeddingsConfig(
        api_base="http://localhost:8080",
        dims=768,
        context_length=2048,
    )
    return LlamaCppServerEmbeddings(config)


class TestLlamaCppEmbeddingFormats:
    """Test various response formats from llama.cpp server"""

    def test_native_format(self, llamacpp_model):
        """Test native llama.cpp format: {"embedding": [floats]}"""
        response = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        result = llamacpp_model._extract_embedding(response)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_array_format(self, llamacpp_model):
        """Test array format: [{"embedding": [floats]}]"""
        response = [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]
        result = llamacpp_model._extract_embedding(response)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_double_nested_array_format(self, llamacpp_model):
        """Test double-nested format: [{"embedding": [[floats]]}]"""
        response = [{"embedding": [[0.1, 0.2, 0.3, 0.4, 0.5]]}]
        result = llamacpp_model._extract_embedding(response)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_openai_compatible_format(self, llamacpp_model):
        """Test OpenAI-compatible format: {"data": [{"embedding": [floats]}]}"""
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0,
                }
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        result = llamacpp_model._extract_embedding(response)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_nested_in_dict_format(self, llamacpp_model):
        """Test nested in dict format: {"embedding": [[floats]]}"""
        response = {"embedding": [[0.1, 0.2, 0.3, 0.4, 0.5]]}
        result = llamacpp_model._extract_embedding(response)
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_invalid_format_raises_error(self, llamacpp_model):
        """Test that invalid format raises ValueError"""
        invalid_responses = [
            {"no_embedding": [0.1, 0.2]},
            [{"no_embedding": [0.1, 0.2]}],
            {"embedding": "not a list"},
            [],
            {},
        ]
        for response in invalid_responses:
            with pytest.raises(ValueError, match="Unsupported embedding response"):
                llamacpp_model._extract_embedding(response)

    @patch("requests.post")
    def test_generate_embedding_with_native_format(self, mock_post, llamacpp_model):
        """Test full generate_embedding method with mocked response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        result = llamacpp_model.generate_embedding("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once_with(
            "http://localhost:8080/embeddings", json={"content": "test text"}
        )

    @patch("requests.post")
    def test_generate_embedding_with_array_format(self, mock_post, llamacpp_model):
        """Test generate_embedding with array response format"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_post.return_value = mock_response

        result = llamacpp_model.generate_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @patch("requests.post")
    def test_generate_embedding_with_openai_format(self, mock_post, llamacpp_model):
        """Test generate_embedding with OpenAI-compatible format"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        }
        mock_post.return_value = mock_response

        result = llamacpp_model.generate_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @patch("requests.post")
    def test_generate_embedding_http_error(self, mock_post, llamacpp_model):
        """Test that HTTP errors are properly raised"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with pytest.raises(Exception):  # requests.HTTPError
            llamacpp_model.generate_embedding("test text")
