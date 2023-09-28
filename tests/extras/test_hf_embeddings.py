"""
Test for HuggingFace embeddings.
This depends on sentence-transformers being installed:
 poetry install -E hf-embeddings
"""


from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig


def test_embeddings():
    sentence_cfg = SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
    )

    sentence_model = EmbeddingModel.create(sentence_cfg)

    sentence_fn = sentence_model.embedding_fn()

    assert len(sentence_fn(["hello"])[0]) == sentence_model.embedding_dims
