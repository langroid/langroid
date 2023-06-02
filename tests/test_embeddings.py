import os

from dotenv import load_dotenv

from llmagent.embedding_models.base import EmbeddingModel
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)


def test_embeddings():
    load_dotenv()
    openai_cfg = OpenAIEmbeddingsConfig(
        model_type="openai",
        model_name="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        dims=1536,
    )

    sentence_cfg = SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        dims=384,
    )

    openai_model = EmbeddingModel.create(openai_cfg)
    sentence_model = EmbeddingModel.create(sentence_cfg)

    openai_fn = openai_model.embedding_fn()
    sentence_fn = sentence_model.embedding_fn()

    # Disable openai test to save on API calls
    assert len(openai_fn(["hello"])[0]) == openai_cfg.dims
    assert len(sentence_fn(["hello"])[0]) == sentence_cfg.dims
