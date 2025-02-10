import os

import pytest
from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import GeminiEmbeddingsConfig


@pytest.mark.skipif(
    os.getenv("GEMINI_API_KEY") is None, reason="GEMINI_API_KEY not set in environment"
)
def test_gemini_embeddings():
    """Test Gemini embedding model for correct output shape."""
    load_dotenv()

    gemini_cfg = GeminiEmbeddingsConfig(model_type="gemini", dims=768)
    gemini_model = EmbeddingModel.create(gemini_cfg)
    gemini_fn = gemini_model.embedding_fn()

    embeddings = gemini_fn(["hello"])  # Returns a List[List[float]]

    assert isinstance(embeddings, list), "Output should be a list"
    assert len(embeddings) == 1, "Should return one embedding for one input"
    assert (
        len(embeddings[0]) == gemini_cfg.dims
    ), f"Expected {gemini_cfg.dims} dims, got {len(embeddings[0])}"
