"""Tests for the Milvus vector-store provider against Milvus Lite."""

import inspect
import json
import math
from typing import List

import pytest

from langroid.embedding_models.base import EmbeddingModel
from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document, Embeddings
from langroid.vector_store.base import VectorStore
from langroid.vector_store.milvusdb import MilvusDB, MilvusDBConfig


class MyDocMetaData(DocMetaData):
    id: str


class MyDoc(Document):
    content: str
    metadata: MyDocMetaData
    category: str = ""


class DeterministicEmbeddingModel(EmbeddingModel):
    @property
    def embedding_dims(self) -> int:
        return 4

    def embedding_fn(self):
        def embed(texts: List[str]) -> Embeddings:
            return [self._embed(text) for text in texts]

        return embed

    @staticmethod
    def _embed(text: str) -> List[float]:
        text = text.lower()
        if "alpha beta" in text:
            return [0.6, 0.8, 0.0, 0.0]
        if "alpha" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in text:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def milvus_lite() -> None:
    pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")


@pytest.fixture(scope="function")
def milvus_vecdb(tmp_path, milvus_lite: None) -> MilvusDB:
    cfg = MilvusDBConfig(
        collection_name="test_milvus",
        uri=str(tmp_path / "milvus.db"),
        replace_collection=True,
        embedding_model=DeterministicEmbeddingModel(),
        batch_size=2,
    )
    try:
        vecdb = VectorStore.create(cfg)
    except LangroidImportError as exc:
        pytest.skip(f"Milvus not installed: {exc}")
    assert isinstance(vecdb, MilvusDB)
    yield vecdb
    vecdb.clear_all_collections(really=True, prefix="test_")
    vecdb.close()


def test_milvus_vector_store_lite_add_search_list_delete_clear(
    milvus_vecdb: MilvusDB,
):
    docs = [
        Document(
            content="alpha document",
            metadata=DocMetaData(
                id="alpha",
                source="group_a",
                note="line1\nline2",
                **{
                    "content": "meta-content",
                    "and": "kw",
                    "tenant-id": "tenant_a",
                    "source.type": "paper",
                },
            ),
        ),
        Document(
            content="beta document",
            metadata=DocMetaData(id="beta", source="group_b"),
        ),
        Document(
            content="gamma document",
            metadata=DocMetaData(id="gamma", source="group_a"),
        ),
    ]
    milvus_vecdb.add_documents(docs)

    assert "test_milvus" in milvus_vecdb.list_collections(empty=True)
    assert "test_milvus" in milvus_vecdb.list_collections()

    all_docs = milvus_vecdb.get_all_documents()
    assert {doc.id() for doc in all_docs} == {"alpha", "beta", "gamma"}

    group_a_docs = milvus_vecdb.get_all_documents(json.dumps({"source": "group_a"}))
    assert {doc.id() for doc in group_a_docs} == {"alpha", "gamma"}

    where = json.dumps({"content": "meta-content"})
    content_docs = milvus_vecdb.get_all_documents(where)
    assert [doc.id() for doc in content_docs] == ["alpha"]
    keyword_docs = milvus_vecdb.get_all_documents(json.dumps({"and": "kw"}))
    assert [doc.id() for doc in keyword_docs] == ["alpha"]
    newline_docs = milvus_vecdb.get_all_documents(json.dumps({"note": "line1\nline2"}))
    assert [doc.id() for doc in newline_docs] == ["alpha"]

    for key, value in {"tenant-id": "tenant_a", "source.type": "paper"}.items():
        punctuation_where = json.dumps({key: value})
        punctuation_docs = milvus_vecdb.get_all_documents(punctuation_where)
        assert [doc.id() for doc in punctuation_docs] == ["alpha"]
        punctuation_matches = milvus_vecdb.similar_texts_with_scores(
            "alpha", k=3, where=punctuation_where
        )
        assert [doc.id() for doc, _ in punctuation_matches] == ["alpha"]

    ordered_docs = milvus_vecdb.get_documents_by_ids(["gamma", "alpha"])
    assert [doc.id() for doc in ordered_docs] == ["gamma", "alpha"]

    docs_and_scores = milvus_vecdb.similar_texts_with_scores("alpha", k=2)
    assert docs_and_scores[0][0].content == "alpha document"
    assert docs_and_scores[0][1] > docs_and_scores[1][1]

    filtered_docs_and_scores = milvus_vecdb.similar_texts_with_scores(
        "alpha",
        k=3,
        where=json.dumps({"source": {"$in": ["group_b"]}}),
    )
    assert [doc.id() for doc, _ in filtered_docs_and_scores] == ["beta"]

    empty_collections = [f"test_empty_{i}" for i in range(2)]
    for collection_name in empty_collections:
        milvus_vecdb.create_collection(collection_name)
    assert milvus_vecdb.clear_empty_collections() == len(empty_collections)

    junk_collections = [f"test_junk_{i}" for i in range(3)]
    for collection_name in junk_collections:
        milvus_vecdb.create_collection(collection_name)
    assert milvus_vecdb.clear_all_collections(
        really=True,
        prefix="test_junk_",
    ) == len(junk_collections)

    milvus_vecdb.delete_collection("test_milvus")
    assert "test_milvus" not in milvus_vecdb.list_collections(empty=True)
    assert milvus_vecdb.get_documents_by_ids(["alpha"]) == []


def test_milvus_unknown_row_count_is_not_empty(
    milvus_vecdb: MilvusDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    milvus_vecdb.add_documents(
        [Document(content="alpha", metadata=DocMetaData(id="alpha"))]
    )
    monkeypatch.setattr(milvus_vecdb.client, "get_collection_stats", lambda **_: {})

    assert milvus_vecdb._row_count("test_milvus") == -1
    assert "test_milvus" in milvus_vecdb.list_collections()
    assert milvus_vecdb.clear_empty_collections() == 0
    assert milvus_vecdb.client.has_collection(collection_name="test_milvus")


def test_milvus_validates_all_documents_before_upsert(
    milvus_vecdb: MilvusDB,
) -> None:
    milvus_vecdb.config.batch_size = 1
    documents = [
        Document(content="alpha", metadata=DocMetaData(id="alpha")),
        Document(
            content="beta",
            metadata=DocMetaData(
                id="x" * (milvus_vecdb.config.id_field_max_length + 1)
            ),
        ),
    ]

    with pytest.raises(ValueError, match="Document id exceeds Milvus max length"):
        milvus_vecdb.add_documents(documents)
    assert milvus_vecdb.get_all_documents() == []


def test_milvus_reuses_loaded_collection_across_clients(
    tmp_path, monkeypatch, milvus_lite: None
):
    uri = str(tmp_path / "milvus_reuse.db")
    collection_name = "test_milvus_reuse"
    first = MilvusDB(
        MilvusDBConfig(
            collection_name=collection_name,
            uri=uri,
            replace_collection=True,
            embedding_model=DeterministicEmbeddingModel(),
        )
    )
    first.add_documents(
        [
            Document(
                content="alpha document",
                metadata=DocMetaData(id="alpha"),
            )
        ]
    )
    first.client.release_collection(collection_name=collection_name)
    first.close()

    second = MilvusDB(
        MilvusDBConfig(
            collection_name=f"{collection_name}_other",
            uri=uri,
            replace_collection=False,
            embedding_model=DeterministicEmbeddingModel(),
        )
    )
    try:
        second.set_collection(collection_name, replace=False)
        assert [doc.id() for doc in second.get_all_documents()] == ["alpha"]
        matches = second.similar_texts_with_scores("alpha", k=1)
        assert [doc.id() for doc, _ in matches] == ["alpha"]

        different_embedding = DeterministicEmbeddingModel()
        monkeypatch.setattr(
            different_embedding,
            "_embed",
            lambda text: [1.0, 0.0, 0.0],
        )
        with pytest.raises(ValueError, match=r"vector dim 4.*embedding dim 3"):
            MilvusDB(
                MilvusDBConfig(
                    collection_name=collection_name,
                    uri=uri,
                    replace_collection=False,
                    embedding_model=different_embedding,
                )
            )
    finally:
        second.clear_all_collections(really=True, prefix="test_milvus_reuse")
        second.close()


def test_milvus_document_subclass_round_trip(milvus_vecdb: MilvusDB):
    milvus_vecdb.config.document_class = MyDoc
    milvus_vecdb.config.metadata_class = MyDocMetaData
    milvus_vecdb.add_documents(
        [
            MyDoc(
                content="alpha document",
                metadata=MyDocMetaData(id="alpha", doc_extra="metadata value"),
                category="research",
            )
        ]
    )
    docs = milvus_vecdb.get_all_documents()

    assert len(docs) == 1
    assert isinstance(docs[0], MyDoc)
    assert docs[0].category == "research"


def test_milvus_lite_version_detection_by_uri():
    milvus_lite_version = MilvusDB._milvus_lite_version()
    server_uris = [
        "unix:/tmp/milvus.sock",
        "tcp://host:19530",
        "grpc://host:19530",
        "http://host:19530",
        "https://x.zillizcloud.com",
    ]
    assert all(not MilvusDB._is_local_lite_uri(uri) for uri in server_uris)
    assert all(not MilvusDB._uses_milvus_lite_3_0(uri) for uri in server_uris)
    assert MilvusDB._is_local_lite_uri("local.db")
    assert MilvusDB._uses_milvus_lite_3_0("local.db") == (
        milvus_lite_version in {"3.0", "3.0.0"}
    )


def test_milvus_config_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("MILVUS_URI", "MILVUS_TOKEN", "MILVUS_DB_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("URI", "leaked-uri")
    monkeypatch.setenv("TOKEN", "leaked-token")
    monkeypatch.setenv("DB_NAME", "leaked-db")

    config = MilvusDBConfig()
    assert (config.uri, config.token, config.db_name) == ("", None, None)

    monkeypatch.setenv("MILVUS_URI", "milvus-uri")
    monkeypatch.setenv("MILVUS_TOKEN", "milvus-token")
    monkeypatch.setenv("MILVUS_DB_NAME", "milvus-db")
    config = MilvusDBConfig()
    assert (config.uri, config.token, config.db_name) == (
        "milvus-uri",
        "milvus-token",
        "milvus-db",
    )

    config = MilvusDBConfig(
        uri="explicit-uri",
        token="explicit-token",
        db_name="explicit-db",
    )
    assert (config.uri, config.token, config.db_name) == (
        "explicit-uri",
        "explicit-token",
        "explicit-db",
    )


def test_milvus_config_default_is_not_shared() -> None:
    # Issue #1079 tracks the repo-wide counterparts.
    config = inspect.signature(MilvusDB.__init__).parameters["config"]
    assert config.default is None


@pytest.mark.parametrize("metric_type", ["COSINE", "IP", "L2"])
def test_milvus_vector_store_lite_score_semantics(
    tmp_path, metric_type: str, milvus_lite: None
):
    cfg = MilvusDBConfig(
        collection_name=f"test_milvus_scores_{metric_type.lower()}",
        uri=str(tmp_path / f"milvus_{metric_type.lower()}.db"),
        metric_type=metric_type,
        replace_collection=True,
        embedding_model=DeterministicEmbeddingModel(),
    )
    try:
        vecdb = VectorStore.create(cfg)
    except LangroidImportError as exc:
        pytest.skip(f"Milvus not installed: {exc}")
    assert isinstance(vecdb, MilvusDB)
    try:
        vecdb.add_documents(
            [
                Document(
                    content="alpha document",
                    metadata=DocMetaData(id="alpha"),
                ),
                Document(
                    content="alpha beta document",
                    metadata=DocMetaData(id="alpha_beta"),
                ),
                Document(
                    content="beta document",
                    metadata=DocMetaData(id="beta"),
                ),
            ]
        )

        docs_and_scores = vecdb.similar_texts_with_scores("alpha", k=3)
        ids = [doc.id() for doc, _ in docs_and_scores]
        scores = [score for _, score in docs_and_scores]

        assert ids == ["alpha", "alpha_beta", "beta"]
        assert scores == sorted(scores, reverse=True)
        if metric_type in ["COSINE", "IP"]:
            assert scores == pytest.approx([1.0, 0.6, 0.0], abs=1e-6)
        else:
            assert scores == pytest.approx(
                [0.0, -math.sqrt(0.8), -math.sqrt(2.0)],
                abs=1e-6,
            )

        if metric_type == "COSINE":
            matches_above_threshold = [
                doc.id() for doc, score in docs_and_scores if score > 0.7
            ]
            assert matches_above_threshold == ["alpha"]
    finally:
        vecdb.clear_all_collections(really=True, prefix="test_milvus_scores_")
        vecdb.close()


@pytest.mark.parametrize(
    "metric_type,raw_score,lite_3_0,expected_score",
    [
        ("COSINE", 0.4, True, 0.6),
        ("COSINE", 0.6, False, 0.6),
        ("IP", 0.6, True, 0.6),
        ("IP", 0.6, False, 0.6),
        ("L2", math.sqrt(0.8), True, -math.sqrt(0.8)),
        ("L2", 0.8, False, -math.sqrt(0.8)),
    ],
)
def test_milvus_score_normalization(
    metric_type: str,
    raw_score: float,
    lite_3_0: bool,
    expected_score: float,
):
    vecdb = object.__new__(MilvusDB)
    vecdb.config = MilvusDBConfig(metric_type=metric_type)
    vecdb._milvus_lite_3_0 = lite_3_0

    assert vecdb._score_from_distance(raw_score) == pytest.approx(expected_score)
