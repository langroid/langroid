"""
Test for Qdrant FastEmbed embeddings, see here:
https://github.com/qdrant/fastembed
This depends on fastembed being installed, either as an extra with langroid, e.g.
     pip install "langroid[fastembed]" (recommended, to get the right version)
or directly via
     pip install fastembed
"""

from langroid.embedding_models.base import EmbeddingModel
from langroid.embedding_models.models import (
    FastEmbedEmbeddingsConfig,
    FastEmbedLateInteractionEmbeddings,
    FastEmbedLateInteractionEmbeddingsConfig,
    FastEmbedSparseEmbeddings,
    FastEmbedSparseEmbeddingsConfig,
)
from langroid.mytypes import SparseEmbedding


def test_embeddings():
    fastembed_cfg = FastEmbedEmbeddingsConfig(
        model_name="BAAI/bge-small-en-v1.5",
    )

    fastembed_model = EmbeddingModel.create(fastembed_cfg)

    fastembed_fn = fastembed_model.embedding_fn()

    assert len(fastembed_fn(["hello"])[0]) == fastembed_model.embedding_dims
    assert fastembed_model.embedding_dims == 384


def test_sparse_embeddings() -> None:
    """Exercise the actual FastEmbed sparse document and query encoders."""
    model = EmbeddingModel.create(
        FastEmbedSparseEmbeddingsConfig(model_name="Qdrant/bm25")
    )
    assert isinstance(model, FastEmbedSparseEmbeddings)
    embeddings = model.embedding_fn()(["materials engineering", "hello world"])
    assert len(embeddings) == 2
    for embedding in embeddings:
        assert isinstance(embedding, SparseEmbedding)
        assert len(embedding.indices) == len(embedding.values) > 0
    assert len(model.query_embedding_fn()(["materials"])) == 1
    assert model.similarity("materials", "materials engineering") > 0


def test_late_interaction_embeddings() -> None:
    """Exercise real token matrices and the separate ColBERT query encoder."""
    model = EmbeddingModel.create(
        FastEmbedLateInteractionEmbeddingsConfig(
            model_name="answerdotai/answerai-colbert-small-v1"
        )
    )
    assert isinstance(model, FastEmbedLateInteractionEmbeddings)
    matrices = model.embedding_fn()(["hello", "a longer passage about materials"])
    assert len(matrices) == 2
    assert all(
        len(row) == model.embedding_dims for matrix in matrices for row in matrix
    )
    assert model.embedding_dims == 96
    queries = model.query_embedding_fn()(["materials"])
    assert len(queries) == 1
    assert all(len(row) == model.embedding_dims for row in queries[0])
    assert model.similarity("materials", "materials engineering") > 0
