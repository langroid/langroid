import os

import pytest
from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import JinaEmbeddingsConfig


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings():
    """Test Jina AI embedding model for correct output shape."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(model_type="jina", dims=1024)
    jina_model = EmbeddingModel.create(jina_cfg)
    jina_fn = jina_model.embedding_fn()

    embeddings = jina_fn(["hello"])  # Returns a List[List[float]]

    assert isinstance(embeddings, list), "Output should be a list"
    assert len(embeddings) == 1, "Should return one embedding for one input"
    assert (
        len(embeddings[0]) == jina_cfg.dims
    ), f"Expected {jina_cfg.dims} dims, got {len(embeddings[0])}"


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings_batch():
    """Test Jina AI embedding model with multiple inputs."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(
        model_type="jina",
        dims=512,  # Test with smaller dimension
        model_name="jina-embeddings-v4",
    )
    jina_model = EmbeddingModel.create(jina_cfg)
    jina_fn = jina_model.embedding_fn()

    texts = ["hello", "world", "test embedding"]
    embeddings = jina_fn(texts)

    assert isinstance(embeddings, list), "Output should be a list"
    assert len(embeddings) == len(texts), f"Should return {len(texts)} embeddings"

    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, list), f"Embedding {i} should be a list"
        assert (
            len(embedding) == jina_cfg.dims
        ), f"Embedding {i}: Expected {jina_cfg.dims} dims, got {len(embedding)}"


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings_similarity():
    """Test that similar texts have higher similarity than dissimilar texts."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(model_type="jina", dims=768)
    jina_model = EmbeddingModel.create(jina_cfg)

    # Test similarity method
    similar_score = jina_model.similarity("cat", "kitten")
    dissimilar_score = jina_model.similarity("cat", "airplane")

    assert (
        similar_score > dissimilar_score
    ), "Similar words should have higher similarity"
    assert 0 <= similar_score <= 1, "Similarity score should be between 0 and 1"
    assert 0 <= dissimilar_score <= 1, "Similarity score should be between 0 and 1"


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings_multimodal():
    """Test Jina AI embedding model with mixed text input formats."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(model_type="jina", dims=768)
    jina_model = EmbeddingModel.create(jina_cfg)
    jina_fn = jina_model.embedding_fn()

    # Test with mixed text input formats (avoid image URLs in unit tests)
    mixed_inputs = [
        "A beautiful sunset",  # Raw text string
        {"text": "Another text example"},  # Pre-formatted text dict
        "Yet another text input",  # Another raw text string
    ]
    
    embeddings = jina_fn(mixed_inputs)

    assert isinstance(embeddings, list), "Output should be a list"
    assert len(embeddings) == len(
        mixed_inputs
    ), f"Should return {len(mixed_inputs)} embeddings"

    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, list), f"Embedding {i} should be a list"
        assert (
            len(embedding) == jina_cfg.dims
        ), f"Embedding {i}: Expected {jina_cfg.dims} dims, got {len(embedding)}"


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings_input_formatting():
    """Test that the input formatting function works correctly."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(model_type="jina")
    jina_model = EmbeddingModel.create(jina_cfg)

    # Test different input types
    assert jina_model._format_input_item("hello") == {"text": "hello"}
    assert jina_model._format_input_item({"text": "hello"}) == {"text": "hello"}
    assert jina_model._format_input_item({"image": "test.jpg"}) == {"image": "test.jpg"}
    assert jina_model._format_input_item("https://example.com/image.jpg") == {
        "image": "https://example.com/image.jpg"
    }


@pytest.mark.skipif(
    os.getenv("JINA_AI_API_KEY") is None,
    reason="JINA_AI_API_KEY not set in environment",
)
def test_jina_embeddings_v4_model():
    """Test Jina AI v4 model with structured input format."""
    load_dotenv()

    jina_cfg = JinaEmbeddingsConfig(
        model_type="jina",
        model_name="jina-embeddings-v4",  # Explicitly test v4
        dims=512,
    )
    jina_model = EmbeddingModel.create(jina_cfg)
    jina_fn = jina_model.embedding_fn()

    # Test with v4 model - should handle both formats
    texts = ["hello world", {"text": "another example"}]
    embeddings = jina_fn(texts)

    assert isinstance(embeddings, list), "Output should be a list"
    assert len(embeddings) == len(texts), f"Should return {len(texts)} embeddings"

    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, list), f"Embedding {i} should be a list"
        assert (
            len(embedding) == jina_cfg.dims
        ), f"Embedding {i}: Expected {jina_cfg.dims} dims, got {len(embedding)}"
