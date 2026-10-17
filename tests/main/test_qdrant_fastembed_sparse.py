"""Offline sparse adapter tests with real in-memory Qdrant.

Fixed encoder outputs isolate conversion and query routing without model
downloads. Real FastEmbed initialization is covered under tests/extras.
"""

import builtins
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client.http.models import NamedVector, SearchRequest, SparseVector

from langroid.embedding_models.base import EmbeddingModel
from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig


class _FixedDenseEmbeddings:
    """Favor the fruit document so a sparse-only hit is distinguishable."""

    embedding_dims = 3

    def embedding_fn(self) -> Callable[[list[str]], list[list[float]]]:
        def encode(texts: list[str]) -> list[list[float]]:
            return [
                [0.0, 1.0, 0.0] if text == "rockets orbit" else [1.0, 0.0, 0.0]
                for text in texts
            ]

        return encode


class _FixedSparseEncoder:
    """Fixture vectors; this does not replace or mock a fastembed module."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, texts: list[str], *, batch_size: int) -> Iterator[SimpleNamespace]:
        self.calls.append(("document", list(texts)))
        vectors = {
            "bananas fruit": ([7], [1.0]),
            "rockets orbit": ([101, 501], [2.0, 0.5]),
            "empty": ([], []),
        }
        for text in texts:
            # A query accidentally routed here produces no matching index.
            indices, values = vectors.get(text, ([999], [1.0]))
            yield SimpleNamespace(
                indices=np.asarray(indices, dtype=np.int64),
                values=np.asarray(values, dtype=np.float32),
            )

    def query_embed(
        self, texts: list[str], *, batch_size: int
    ) -> Iterator[SimpleNamespace]:
        self.calls.append(("query", list(texts)))
        for text in texts:
            assert text == "launch"
            yield SimpleNamespace(
                indices=np.asarray([101], dtype=np.int64),
                values=np.asarray([3.0], dtype=np.float32),
            )


@pytest.fixture
def fixed_dense_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    dense = _FixedDenseEmbeddings()
    monkeypatch.setattr(EmbeddingModel, "create", staticmethod(lambda config: dense))


@pytest.fixture
def sparse_db(fixed_dense_embeddings: None) -> Iterator[QdrantDB]:
    config = QdrantDBConfig(
        cloud=False,
        storage_path=":memory:",
        collection_name=None,
        use_sparse_embeddings=False,
        sparse_embedding_backend="fastembed",
    )
    with QdrantDB(config) as db:
        # Install deterministic output at the encoder boundary. The real
        # constructor/model path is exercised separately in tests/extras.
        db.sparse_fastembed_model = cast(Any, _FixedSparseEncoder())
        db.config.use_sparse_embeddings = True
        db.create_collection("offline-sparse")
        yield db


def test_existing_sparse_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "QDRANT_USE_SPARSE_EMBEDDINGS",
        "QDRANT_SPARSE_EMBEDDING_BACKEND",
        "QDRANT_SPARSE_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    config = QdrantDBConfig()
    assert config.use_sparse_embeddings is False
    assert config.sparse_embedding_backend == "transformers"
    assert config.sparse_embedding_model == "naver/splade-v3-distilbert"


def test_invalid_sparse_backend_is_rejected() -> None:
    with pytest.raises(ValidationError, match="sparse_embedding_backend"):
        QdrantDBConfig(sparse_embedding_backend="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["fastembed", "transformers"])
def test_missing_backend_dependency_has_install_guidance(
    backend: str,
    fixed_dense_embeddings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == backend:
            raise ImportError(f"{backend} is unavailable in this test")
        return original_import(name, *args, **kwargs)

    config = QdrantDBConfig(
        cloud=False,
        storage_path=":memory:",
        collection_name=None,
        use_sparse_embeddings=True,
        sparse_embedding_backend=backend,  # type: ignore[arg-type]
    )
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(ImportError, match=rf"langroid\[{backend}\]"):
            QdrantDB(config)


def test_disabled_sparse_does_not_initialize_fastembed(
    fixed_dense_embeddings: None,
) -> None:
    config = QdrantDBConfig(
        cloud=False,
        storage_path=":memory:",
        collection_name="dense-only",
        use_sparse_embeddings=False,
        sparse_embedding_backend="fastembed",
    )
    with QdrantDB(config) as db:
        assert db.sparse_fastembed_model is None
        assert db.get_sparse_embeddings(["text"]) == []
        info = db.client.get_collection("dense-only")
        assert not info.config.params.sparse_vectors
        db.add_documents([Document(metadata=DocMetaData(), content="bananas fruit")])
        assert db.similar_texts_with_scores("launch", k=1)[0][0].content == (
            "bananas fruit"
        )


@pytest.mark.parametrize("is_query", [False, True])
def test_empty_input_does_not_invoke_encoder(
    sparse_db: QdrantDB, is_query: bool
) -> None:
    encoder = cast(_FixedSparseEncoder, sparse_db.sparse_fastembed_model)
    assert sparse_db.get_sparse_embeddings([], is_query=is_query) == []
    assert encoder.calls == []


def test_sparse_vectors_preserve_indices_weights_and_query_routing(
    sparse_db: QdrantDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def no_torch_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError("FastEmbed sparse encoding must not import torch")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_torch_import)
        documents = sparse_db.get_sparse_embeddings(["bananas fruit", "rockets orbit"])
        queries = sparse_db.get_sparse_embeddings(["launch"], is_query=True)
    assert all(isinstance(vector, SparseVector) for vector in documents + queries)
    assert documents[0].indices == [7]
    assert documents[0].values == [1.0]
    assert documents[1].indices == [101, 501]
    assert documents[1].values == [2.0, 0.5]
    assert queries[0].indices == [101]
    assert queries[0].values == [3.0]


def test_sparse_indexing_and_search_use_real_in_memory_qdrant(
    sparse_db: QdrantDB,
) -> None:
    sparse_db.add_documents(
        [
            Document(metadata=DocMetaData(), content="bananas fruit"),
            Document(metadata=DocMetaData(), content="rockets orbit"),
        ]
    )
    dense_hits = sparse_db.client.search_batch(
        collection_name="offline-sparse",
        requests=[
            SearchRequest(
                vector=NamedVector(name="", vector=[1.0, 0.0, 0.0]),
                limit=1,
                with_payload=True,
            )
        ],
    )[0]
    assert dense_hits[0].payload is not None
    assert dense_hits[0].payload["content"] == "bananas fruit"

    # Dense search alone cannot return the rocket document at k=1.
    results = sparse_db.similar_texts_with_scores("launch", k=1)
    rocket_scores = [score for doc, score in results if doc.content == "rockets orbit"]
    assert rocket_scores == pytest.approx([6.0])
    encoder = cast(_FixedSparseEncoder, sparse_db.sparse_fastembed_model)
    assert ("query", ["launch"]) in encoder.calls


def test_empty_sparse_vector_can_be_stored(sparse_db: QdrantDB) -> None:
    vector = sparse_db.get_sparse_embeddings(["empty"])[0]
    assert vector.indices == []
    assert vector.values == []
    sparse_db.add_documents([Document(metadata=DocMetaData(), content="empty")])
    assert [doc.content for doc in sparse_db.get_all_documents()] == ["empty"]
