"""
Test for HuggingFace embeddings.
This depends on sentence-transformers being installed:
uv sync --dev --extra hf-embeddings
"""

import os

import pytest

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig
from langroid.embedding_models.remote_embeds import RemoteEmbeddingsConfig


def test_embeddings():
    sentence_cfg = SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    sentence_model = EmbeddingModel.create(sentence_cfg)

    sentence_fn = sentence_model.embedding_fn()

    assert len(sentence_fn(["hello"])[0]) == sentence_model.embedding_dims


# skip this if CI is true in env
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Fine locally but fails in GH CI"
)
def test_remote_embeddings():
    sentence_cfg = RemoteEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    sentence_model = EmbeddingModel.create(sentence_cfg)

    sentence_fn = sentence_model.embedding_fn()

    assert len(sentence_fn(["hello"])[0]) == sentence_model.embedding_dims
