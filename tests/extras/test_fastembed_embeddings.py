"""
Test for HuggingFace embeddings.
This depends on fastembed being installed:
 poetry install fastembed
"""

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import FastEmbedEmbeddingsConfig


def test_embeddings():
    fastembed_cfg = FastEmbedEmbeddingsConfig(
        model_name= "BAAI/bge-small-en-v1.5",
        dims=384,
    )

    fastembed_model = EmbeddingModel.create(fastembed_cfg)

    fastembed_fn = fastembed_model.embedding_fn()

    assert len(fastembed_fn(["hello"])[0]) == fastembed_model.embedding_dims

