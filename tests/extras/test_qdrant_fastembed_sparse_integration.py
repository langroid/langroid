"""Real FastEmbed SPLADE integration; downloads a model on first execution.

Run explicitly with the fastembed extra. No Qdrant server or API key is needed.
"""

import builtins
from collections.abc import Callable
from typing import Any

import pytest
from qdrant_client.http.models import SparseVector

from langroid.embedding_models.base import EmbeddingModel
from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig


class _FixedDenseEmbeddings:
    """Keep the dense side offline and favor the unrelated document."""

    embedding_dims = 3

    def embedding_fn(self) -> Callable[[list[str]], list[list[float]]]:
        def encode(texts: list[str]) -> list[list[float]]:
            return [
                (
                    [0.0, 1.0, 0.0]
                    if "rocket" in text and text != "rocket"
                    else [1.0, 0.0, 0.0]
                )
                for text in texts
            ]

        return encode


def test_real_fastembed_splade_indexing_and_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastembed")
    dense = _FixedDenseEmbeddings()
    monkeypatch.setattr(EmbeddingModel, "create", staticmethod(lambda config: dense))

    config = QdrantDBConfig(
        cloud=False,
        storage_path=":memory:",
        collection_name="real-fastembed-sparse",
        use_sparse_embeddings=True,
        sparse_embedding_backend="fastembed",
        fastembed_sparse_model="prithivida/Splade_PP_en_v1",
    )
    original_import = builtins.__import__

    def unavailable_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"torch", "transformers"}:
            raise ImportError("torch and transformers are unavailable in this test")
        return original_import(name, *args, **kwargs)

    # Keep FastEmbed itself real, including model loading and inference.
    # Blocking imports checks that this sparse path works without torch.
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", unavailable_torch)
        with QdrantDB(config) as db:
            assert db.sparse_fastembed_model is not None
            contents = [
                "Fresh bananas and apples are sold at the fruit market.",
                "The rocket launches a satellite into orbit in outer space.",
            ]
            vectors = db.get_sparse_embeddings(contents)
            assert len(vectors) == len(contents)
            for vector in vectors:
                assert isinstance(vector, SparseVector)
                assert vector.indices
                assert len(vector.indices) == len(vector.values)
                assert all(value > 0 for value in vector.values)

            query = db.get_sparse_embeddings(["rocket"], is_query=True)
            assert len(query) == 1
            assert query[0].indices

            db.add_documents(
                [Document(metadata=DocMetaData(), content=text) for text in contents]
            )
            assert len(db.get_all_documents()) == 2
            results = db.similar_texts_with_scores("rocket", k=1)
            assert any(
                doc.content == contents[1] and score > 0 for doc, score in results
            )

            records, _ = db.client.scroll(
                collection_name="real-fastembed-sparse",
                with_vectors=True,
                with_payload=True,
                limit=10,
            )
            assert len(records) == 2
            for record in records:
                assert isinstance(record.vector, dict)
                stored = record.vector["text-sparse"]
                assert isinstance(stored, SparseVector)
                assert stored.indices
                assert len(stored.indices) == len(stored.values)
