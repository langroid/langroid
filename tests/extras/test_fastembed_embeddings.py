"""
Test for Qdrant FastEmbed embeddings, see here:
https://github.com/qdrant/fastembed
This depends on fastembed being installed, either as an extra with langroid, e.g.
     pip install "langroid[fastembed]" (recommended, to get the right version)
or directly via
     pip install fastembed
"""

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import FastEmbedEmbeddingsConfig


def test_embeddings():
    fastembed_cfg = FastEmbedEmbeddingsConfig(
        model_name="BAAI/bge-small-en-v1.5",
    )

    fastembed_model = EmbeddingModel.create(fastembed_cfg)

    fastembed_fn = fastembed_model.embedding_fn()

    assert len(fastembed_fn(["hello"])[0]) == fastembed_model.embedding_dims
    assert fastembed_model.embedding_dims == 384
