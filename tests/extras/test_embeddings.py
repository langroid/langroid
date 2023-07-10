from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig

# this depends on sentence-transformers being installed
# poetry install -E hf-embeddings


def test_embeddings():
    sentence_cfg = SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        dims=384,
    )

    sentence_model = EmbeddingModel.create(sentence_cfg)

    sentence_fn = sentence_model.embedding_fn()

    assert len(sentence_fn(["hello"])[0]) == sentence_cfg.dims
