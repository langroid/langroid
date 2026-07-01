"""
Unit tests for the Dakera vector store, using a mocked Dakera client and a fake
embedding function so they run without a live server or embedding API.

Integration coverage (against a real Dakera server) is provided by the
parametrized suite in ``tests/main/test_vector_stores.py`` when a server and
``DAKERA_API_KEY`` are available.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("dakera")

from dakera import DistanceMetric, Vector  # noqa: E402

from langroid.mytypes import DocMetaData, Document  # noqa: E402
from langroid.vector_store.dakera import DakeraDB, DakeraDBConfig  # noqa: E402

CLIENT_PATH = "langroid.vector_store.dakera.DakeraClient"
EMB_CREATE_PATH = "langroid.vector_store.base.EmbeddingModel.create"


class _FakeEmbeddingModel:
    """Deterministic 3-dim embeddings, no network."""

    def embedding_fn(self):
        return lambda texts: [[0.1, 0.2, 0.3] for _ in texts]


def _make_db(mock_client_cls, collection_name="test-coll"):
    client = mock_client_cls.return_value
    client.list_namespaces.return_value = []
    with patch(EMB_CREATE_PATH, return_value=_FakeEmbeddingModel()):
        db = DakeraDB(
            DakeraDBConfig(collection_name=collection_name, api_key="dk-fake")
        )
    return db, client


@patch(CLIENT_PATH)
def test_init_configures_namespace(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    mock_client_cls.assert_called_once_with(
        base_url="http://localhost:3000", api_key="dk-fake"
    )
    # embedding_dim is inferred from the fake embedding function (len == 3).
    client.configure_namespace.assert_called_once()
    kwargs = client.configure_namespace.call_args.kwargs
    assert kwargs["dimension"] == 3
    assert kwargs["distance"] == DistanceMetric.COSINE


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("DAKERA_API_KEY", raising=False)
    with patch(CLIENT_PATH), patch(EMB_CREATE_PATH, return_value=_FakeEmbeddingModel()):
        with pytest.raises(ValueError, match="DAKERA_API_KEY"):
            DakeraDB(DakeraDBConfig(collection_name=None, api_key=""))


@patch(CLIENT_PATH)
def test_add_documents_and_roundtrip_via_query(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    # Collection already exists (non-empty check in add_documents).
    client.list_namespaces.return_value = [
        SimpleNamespace(name="test-coll", vector_count=0)
    ]

    docs = [
        Document(content="hello world", metadata=DocMetaData()),
        Document(content="goodbye", metadata=DocMetaData()),
    ]
    db.add_documents(docs)

    client.upsert.assert_called_once()
    ns_arg = client.upsert.call_args.args[0]
    vectors = client.upsert.call_args.kwargs["vectors"]
    assert ns_arg == "test-coll"
    assert len(vectors) == 2
    assert vectors[0].values == [0.1, 0.2, 0.3]
    assert vectors[0].metadata["content"] == "hello world"

    # Round-trip: feed the stored metadata back through a query and confirm the
    # Document is reconstructed with the same content and score.
    stored_meta = vectors[0].metadata
    client.query.return_value = SimpleNamespace(
        results=[SimpleNamespace(id="x", score=0.88, values=None, metadata=stored_meta)]
    )
    pairs = db.similar_texts_with_scores("hello", k=3)
    assert client.query.call_args.kwargs["top_k"] == 3
    assert client.query.call_args.kwargs["distance_metric"] == DistanceMetric.COSINE
    assert len(pairs) == 1
    doc, score = pairs[0]
    assert doc.content == "hello world"
    assert score == 0.88


@patch(CLIENT_PATH)
def test_get_documents_by_ids_preserves_order(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    client.list_namespaces.return_value = [
        SimpleNamespace(name="test-coll", vector_count=0)
    ]
    db.add_documents([Document(content="alpha", metadata=DocMetaData())])
    stored_meta = client.upsert.call_args.kwargs["vectors"][0].metadata

    client.fetch.return_value = [
        Vector(id="b", values=[0.0, 0.0, 0.0], metadata=stored_meta),
        Vector(id="a", values=[0.0, 0.0, 0.0], metadata=stored_meta),
    ]
    docs = db.get_documents_by_ids(["a", "b", "missing"])
    # Order follows the requested ids; missing ids are skipped.
    assert [d.content for d in docs] == ["alpha", "alpha"]
    assert client.fetch.call_args.args[0] == "test-coll"


@patch(CLIENT_PATH)
def test_get_documents_by_ids_empty(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    assert db.get_documents_by_ids([]) == []
    client.fetch.assert_not_called()


@patch(CLIENT_PATH)
def test_list_collections_filters_empty(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    client.list_namespaces.return_value = [
        SimpleNamespace(name="a", vector_count=2),
        SimpleNamespace(name="b", vector_count=0),
    ]
    assert db.list_collections(empty=False) == ["a"]
    assert set(db.list_collections(empty=True)) == {"a", "b"}


@patch(CLIENT_PATH)
def test_clear_empty_collections(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    client.list_namespaces.return_value = [
        SimpleNamespace(name="a", vector_count=2),
        SimpleNamespace(name="b", vector_count=0),
        SimpleNamespace(name="c", vector_count=0),
    ]
    assert db.clear_empty_collections() == 2
    deleted = {c.args[0] for c in client.delete_namespace.call_args_list}
    assert deleted == {"b", "c"}


@patch(CLIENT_PATH)
def test_clear_all_collections_requires_really(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    assert db.clear_all_collections(really=False) == 0
    client.delete_namespace.assert_not_called()


@patch(CLIENT_PATH)
def test_clear_all_collections_with_prefix(mock_client_cls):
    db, client = _make_db(mock_client_cls)
    client.list_namespaces.return_value = [
        SimpleNamespace(name="keep-1", vector_count=1),
        SimpleNamespace(name="drop-1", vector_count=1),
        SimpleNamespace(name="drop-2", vector_count=0),
    ]
    assert db.clear_all_collections(really=True, prefix="drop-") == 2
    deleted = {c.args[0] for c in client.delete_namespace.call_args_list}
    assert deleted == {"drop-1", "drop-2"}


@patch(CLIENT_PATH)
def test_similar_texts_rejects_bad_k(mock_client_cls):
    db, _ = _make_db(mock_client_cls)
    with pytest.raises(ValueError, match="must be >= 1"):
        db.similar_texts_with_scores("q", k=0)


@patch(CLIENT_PATH)
def test_metric_defaults_to_cosine_for_unknown(mock_client_cls):
    client = mock_client_cls.return_value
    client.list_namespaces.return_value = []
    with patch(EMB_CREATE_PATH, return_value=_FakeEmbeddingModel()):
        db = DakeraDB(
            DakeraDBConfig(collection_name="c", api_key="dk-fake", metric="manhattan")
        )
    assert db._distance() == DistanceMetric.COSINE
