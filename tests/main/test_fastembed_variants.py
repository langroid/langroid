"""FastEmbed adapter regressions without downloads or optional dependencies."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Type
from unittest.mock import MagicMock

import numpy as np
import pytest

from langroid.embedding_models import (
    EmbeddingModel,
    FastEmbedEmbeddings,
    FastEmbedEmbeddingsConfig,
    FastEmbedLateInteractionEmbeddings,
    FastEmbedLateInteractionEmbeddingsConfig,
    FastEmbedSparseEmbeddings,
    FastEmbedSparseEmbeddingsConfig,
)
from langroid.embedding_models.models import embedding_model
from langroid.exceptions import LangroidImportError
from langroid.mytypes import SparseEmbedding
from langroid.vector_store.base import VectorStore, VectorStoreConfig


@pytest.fixture
def fastembed(monkeypatch: pytest.MonkeyPatch) -> Dict[str, MagicMock]:
    module = ModuleType("fastembed")
    constructors = {}
    for name in (
        "TextEmbedding",
        "SparseTextEmbedding",
        "LateInteractionTextEmbedding",
    ):
        constructors[name] = MagicMock()
        setattr(module, name, constructors[name])
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return constructors


@pytest.mark.parametrize(
    "config_type, model_type, backend, selector",
    [
        (FastEmbedEmbeddingsConfig, FastEmbedEmbeddings, "TextEmbedding", "fastembed"),
        (
            FastEmbedSparseEmbeddingsConfig,
            FastEmbedSparseEmbeddings,
            "SparseTextEmbedding",
            "fastembed-sparse",
        ),
        (
            FastEmbedLateInteractionEmbeddingsConfig,
            FastEmbedLateInteractionEmbeddings,
            "LateInteractionTextEmbedding",
            "fastembed-late-interaction",
        ),
    ],
)
def test_factory_and_config_forwarding(
    fastembed: Dict[str, MagicMock],
    config_type: Type[FastEmbedEmbeddingsConfig],
    model_type: Type[EmbeddingModel],
    backend: str,
    selector: str,
) -> None:
    config = config_type(
        model_name="test-model",
        cache_dir="cache",
        threads=2,
        batch_size=7,
        parallel=3,
        additional_kwargs={"providers": ["CPUExecutionProvider"]},
    )
    model = EmbeddingModel.create(config)
    assert isinstance(model, model_type)
    assert embedding_model(selector) is model_type
    fastembed[backend].assert_called_once_with(
        model_name="test-model",
        cache_dir="cache",
        threads=2,
        providers=["CPUExecutionProvider"],
    )
    cloned = model.clone()
    assert isinstance(cloned, model_type)
    assert cloned is not model
    assert cloned.config is not config
    assert cloned.config == config


def test_dense_behavior_unchanged(fastembed: Dict[str, MagicMock]) -> None:
    backend = fastembed["TextEmbedding"].return_value
    backend.embed.side_effect = lambda texts, **kw: (
        np.array([1.0, 2.0, 3.0]) for text in texts
    )
    model = FastEmbedEmbeddings()
    assert model.embedding_type == "dense"
    assert model.embedding_fn()(["first", "second"]) == [[1.0, 2.0, 3.0]] * 2
    assert model.query_embedding_fn()(["query"]) == [[1.0, 2.0, 3.0]]
    assert model.embedding_dims == 3
    assert model.similarity("first", "second") == pytest.approx(1.0)
    store = MagicMock()
    VectorStore.__init__(store, VectorStoreConfig(embedding_model=model))
    assert store.embedding_fn(["document"]) == [[1.0, 2.0, 3.0]]


def test_sparse_outputs_and_query_encoder(fastembed: Dict[str, MagicMock]) -> None:
    backend = fastembed["SparseTextEmbedding"].return_value
    raw = [
        SimpleNamespace(
            indices=np.array([1, 1_000_000_000]), values=np.array([2.0, 3.0])
        ),
        SimpleNamespace(indices=np.array([4]), values=np.array([5.0])),
    ]
    backend.passage_embed.side_effect = lambda *a, **kw: iter(raw)
    backend.query_embed.side_effect = lambda *a, **kw: iter(raw[:1])
    model = FastEmbedSparseEmbeddings(
        FastEmbedSparseEmbeddingsConfig(batch_size=7, parallel=3)
    )
    output = model.embedding_fn()(["first", "second"])
    assert output == [
        SparseEmbedding(indices=[1, 1_000_000_000], values=[2.0, 3.0]),
        SparseEmbedding(indices=[4], values=[5.0]),
    ]
    assert SparseEmbedding.model_validate_json(output[0].model_dump_json()) == output[0]
    backend.passage_embed.assert_called_once_with(
        ["first", "second"], batch_size=7, parallel=3
    )
    assert model.query_embedding_fn()(["query"]) == output[:1]
    backend.query_embed.assert_called_once_with(["query"], batch_size=7, parallel=3)
    with pytest.raises(ValueError, match="fixed dense dimension"):
        _ = model.embedding_dims


def test_sparse_similarity(fastembed: Dict[str, MagicMock]) -> None:
    backend = fastembed["SparseTextEmbedding"].return_value
    backend.query_embed.return_value = [
        SimpleNamespace(indices=np.array([1, 100]), values=np.array([2.0, 3.0]))
    ]
    backend.passage_embed.return_value = [
        SimpleNamespace(indices=np.array([4, 100]), values=np.array([4.0, 5.0]))
    ]
    model = FastEmbedSparseEmbeddings()
    assert model.similarity("query", "document") == 15.0
    backend.query_embed.return_value = [
        SimpleNamespace(indices=np.array([], dtype=int), values=np.array([]))
    ]
    assert model.similarity("empty", "document") == 0.0


def test_late_interaction_preserves_matrices(
    fastembed: Dict[str, MagicMock],
) -> None:
    backend = fastembed["LateInteractionTextEmbedding"].return_value
    matrices = [np.eye(2, 3), np.ones((4, 3))]
    backend.passage_embed.side_effect = lambda texts, **kw: iter(matrices[: len(texts)])
    backend.query_embed.return_value = iter([np.ones((1, 3))])
    model = FastEmbedLateInteractionEmbeddings(
        FastEmbedLateInteractionEmbeddingsConfig(batch_size=7, parallel=3)
    )
    assert model.embedding_fn()(["short", "long"]) == [m.tolist() for m in matrices]
    backend.passage_embed.assert_called_once_with(
        ["short", "long"], batch_size=7, parallel=3
    )
    assert model.query_embedding_fn()(["query"]) == [[[1.0, 1.0, 1.0]]]
    backend.query_embed.assert_called_once_with(["query"], batch_size=7, parallel=3)
    assert model.embedding_dims == 3  # Token width, not the two-token length.
    calls = backend.passage_embed.call_count
    assert model.embedding_dims == 3
    assert backend.passage_embed.call_count == calls


def test_late_interaction_maxsim(fastembed: Dict[str, MagicMock]) -> None:
    backend = fastembed["LateInteractionTextEmbedding"].return_value
    backend.query_embed.return_value = [np.eye(2, 3)]
    backend.passage_embed.return_value = [
        np.array([[0.5, 0.5, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ]
    model = FastEmbedLateInteractionEmbeddings()
    assert model.similarity("query", "document") == 1.5
    backend.passage_embed.return_value = [np.empty((0, 3))]
    assert model.similarity("query", "empty") == 0.0


@pytest.mark.parametrize(
    "model_type, backend",
    [
        (FastEmbedSparseEmbeddings, "SparseTextEmbedding"),
        (FastEmbedLateInteractionEmbeddings, "LateInteractionTextEmbedding"),
    ],
)
def test_empty_input_and_dense_store_guard(
    fastembed: Dict[str, MagicMock],
    model_type: Type[EmbeddingModel],
    backend: str,
) -> None:
    model = model_type()
    assert model.embedding_fn()([]) == []
    assert model.query_embedding_fn()([]) == []
    fastembed[backend].return_value.passage_embed.assert_not_called()
    fastembed[backend].return_value.query_embed.assert_not_called()
    with pytest.raises(ValueError, match="requires dense embeddings"):
        VectorStore.__init__(MagicMock(), VectorStoreConfig(embedding_model=model))
    fastembed[backend].return_value.passage_embed.assert_not_called()


@pytest.mark.parametrize(
    "model_type", [FastEmbedSparseEmbeddings, FastEmbedLateInteractionEmbeddings]
)
def test_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, model_type: Type[EmbeddingModel]
) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(LangroidImportError, match="fastembed"):
        model_type()


def test_empty_token_dimension(fastembed: Dict[str, MagicMock]) -> None:
    fastembed[
        "LateInteractionTextEmbedding"
    ].return_value.passage_embed.return_value = [np.empty((0, 3))]
    with pytest.raises(ValueError, match="empty token matrix"):
        _ = FastEmbedLateInteractionEmbeddings().embedding_dims
