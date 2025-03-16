import pytest
from dotenv import find_dotenv, load_dotenv

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import (
    AzureOpenAIEmbeddingsConfig,
    OpenAIEmbeddingsConfig,
)


def test_openai_embeddings():
    load_dotenv(find_dotenv(usecwd=True))
    openai_cfg = OpenAIEmbeddingsConfig(
        model_type="openai",
        model_name="text-embedding-3-small",
        dims=1536,
    )

    openai_model = EmbeddingModel.create(openai_cfg)

    openai_fn = openai_model.embedding_fn()

    assert len(openai_fn(["hello"])[0]) == openai_cfg.dims


def test_azure_openai_embeddings():
    load_dotenv(find_dotenv(usecwd=True))
    azure_openai_cfg = AzureOpenAIEmbeddingsConfig(
        model_type="azure-openai",
        model_name="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        dims=1536,
    )
    azure_openai_model = EmbeddingModel.create(azure_openai_cfg)

    azure_openai_fn = azure_openai_model.embedding_fn()

    assert len(azure_openai_fn(["hello"])[0]) == azure_openai_cfg.dims


@pytest.mark.xfail(
    reason="LangDB may fail due to unknown flakiness",
    run=True,
    strict=False,
)
def test_langdb_embeddings():
    """Test that embedding models work via LangDB"""
    langdb_openai_embed_config = OpenAIEmbeddingsConfig(
        model_name="langdb/openai/text-embedding-3-small",
    )
    langdb_openai_embed_model = EmbeddingModel.create(langdb_openai_embed_config)
    emb_fn = langdb_openai_embed_model.embedding_fn()
    assert len(emb_fn(["hello"])[0]) == langdb_openai_embed_config.dims
