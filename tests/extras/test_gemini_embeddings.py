from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import GeminiEmbeddingsConfig


def test_gemini_embeddings():
    load_dotenv()
    gemini_cfg = GeminiEmbeddingsConfig(model_type="gemini", dims=768)

    gemini_model = EmbeddingModel.create(gemini_cfg)

    gemini_fn = gemini_model.embedding_fn()

    assert len(gemini_fn(["hello"])[0]) == gemini_cfg.dims
