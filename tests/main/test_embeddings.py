from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import OpenAIEmbeddingsConfig


def test_embeddings():
    load_dotenv()
    openai_cfg = OpenAIEmbeddingsConfig(
        model_type="openai",
        model_name="text-embedding-ada-002",
        dims=1536,
    )

    openai_model = EmbeddingModel.create(openai_cfg)

    openai_fn = openai_model.embedding_fn()

    assert len(openai_fn(["hello"])[0]) == openai_cfg.dims
