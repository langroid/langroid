from types import SimpleNamespace
from typing import List

import pytest
from dotenv import load_dotenv

from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.meilisearch import MeiliSearch, MeiliSearchConfig
from langroid.vector_store.momento import MomentoVI, MomentoVIConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

load_dotenv()
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

phrases = SimpleNamespace(
    HELLO="hello",
    HI_THERE="hi there",
    CANADA="people living in Canada",
    NOT_CANADA="people not living in Canada",
    OVER_40="people over 40",
    UNDER_40="people under 40",
    FRANCE="what is the capital of France?",
    BELGIUM="which city is Belgium's capital?",
)


class MyDocMetaData(DocMetaData):
    id: str


class MyDoc(Document):
    content: str
    metadata: MyDocMetaData


stored_docs = [
    MyDoc(content=d, metadata=MyDocMetaData(id=str(i)))
    for i, d in enumerate(vars(phrases).values())
]


@pytest.fixture(scope="function")
def vecdb(request) -> VectorStore:
    if request.param == "qdrant_local":
        qd_dir = ".qdrant/data/" + embed_cfg.model_type
        rmdir(qd_dir)
        qd_cfg = QdrantDBConfig(
            cloud=False,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=qd_dir,
            embedding=embed_cfg,
        )
        qd = QdrantDB(qd_cfg)
        qd.add_documents(stored_docs)
        yield qd
        rmdir(qd_dir)
        return

    if request.param == "qdrant_cloud":
        qd_dir = ".qdrant/cloud/" + embed_cfg.model_type
        qd_cfg_cloud = QdrantDBConfig(
            cloud=True,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=qd_dir,
            embedding=embed_cfg,
        )
        qd_cloud = QdrantDB(qd_cfg_cloud)
        qd_cloud.add_documents(stored_docs)
        yield qd_cloud
        qd_cloud.delete_collection(collection_name=qd_cfg_cloud.collection_name)
        return

    if request.param == "chroma":
        cd_dir = ".chroma/" + embed_cfg.model_type
        rmdir(cd_dir)
        cd_cfg = ChromaDBConfig(
            collection_name="test-" + embed_cfg.model_type,
            storage_path=cd_dir,
            embedding=embed_cfg,
        )
        cd = ChromaDB(cd_cfg)
        cd.add_documents(stored_docs)
        yield cd
        rmdir(cd_dir)
        return

    if request.param == "meilisearch":
        ms_cfg = MeiliSearchConfig(
            collection_name="test-meilisearch",
        )
        ms = MeiliSearch(ms_cfg)
        ms.add_documents(stored_docs)
        yield ms
        ms.delete_collection(collection_name=ms_cfg.collection_name)
        return

    if request.param == "momento":
        cfg = MomentoVIConfig(
            collection_name="test-momento",
        )
        vdb = MomentoVI(cfg)
        vdb.add_documents(stored_docs)
        yield vdb
        vdb.delete_collection(collection_name=cfg.collection_name)

    if request.param == "lancedb":
        ldb_dir = ".lancedb/data/" + embed_cfg.model_type
        rmdir(ldb_dir)
        ldb_cfg = LanceDBConfig(
            cloud=False,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=ldb_dir,
            embedding=embed_cfg,
            document_class=MyDoc,  # IMPORTANT, to ensure table has full schema!
        )
        ldb = LanceDB(ldb_cfg)
        ldb.add_documents(stored_docs)
        yield ldb
        rmdir(ldb_dir)
        return


@pytest.mark.parametrize(
    "query,results,exceptions",
    [
        ("which city is Belgium's capital?", [phrases.BELGIUM], ["meliseach"]),
        ("capital of France", [phrases.FRANCE], ["meliseach"]),
        ("hello", [phrases.HELLO], ["meliseach"]),
        ("hi there", [phrases.HI_THERE], ["meliseach"]),
        ("men and women over 40", [phrases.OVER_40], ["meilisearch"]),
        ("people aged less than 40", [phrases.UNDER_40], ["meilisearch"]),
        ("Canadian residents", [phrases.CANADA], ["meilisearch"]),
        ("people outside Canada", [phrases.NOT_CANADA], ["meilisearch"]),
    ],
)
# add "momento" when index-creation timeout error is resolved.
@pytest.mark.parametrize(
    "vecdb",
    ["lancedb", "chroma", "meilisearch", "qdrant_local", "qdrant_cloud"],
    indirect=True,
)
def test_vector_stores_search(
    vecdb, query: str, results: List[str], exceptions: List[str]
):
    if vecdb.__class__.__name__.lower() in exceptions:
        # we don't expect some of these to work,
        # e.g. MeiliSearch is a text search engine, not a vector store
        return
    if isinstance(vecdb, MomentoVI):
        # skip due to non-deterministic search failures. Maybe need to use async?
        return
    docs_and_scores = vecdb.similar_texts_with_scores(query, k=len(vars(phrases)))
    # first doc should be best match
    if isinstance(vecdb, ChromaDB):
        # scores are (apparently) l2 distances (docs unclear), so low means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score < 0.3]
    else:
        # scores are cosine similarities, so high means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score > 0.7]
    assert set(results).issubset(set(matching_docs))


# add "momento" when index-creation timeout error is resolved.
@pytest.mark.parametrize(
    "vecdb",
    ["lancedb", "meilisearch", "chroma", "qdrant_local", "qdrant_cloud"],
    indirect=True,
)
def test_vector_stores_access(vecdb):
    assert vecdb is not None

    if not isinstance(vecdb, MomentoVI):
        all_docs = vecdb.get_all_documents()
        assert len(all_docs) == len(stored_docs)

    coll_name = vecdb.config.collection_name
    assert coll_name is not None

    vecdb.delete_collection(collection_name=coll_name)
    vecdb.create_collection(collection_name=coll_name)
    if not isinstance(vecdb, MomentoVI):
        all_docs = vecdb.get_all_documents()
        assert len(all_docs) == 0

    if isinstance(vecdb, MomentoVI):
        return

    vecdb.add_documents(stored_docs)
    all_docs = vecdb.get_all_documents()
    ids = [doc.id() for doc in all_docs]
    assert len(all_docs) == len(stored_docs)

    docs = vecdb.get_documents_by_ids(ids[:3])
    assert len(docs) == 3

    coll_names = [f"test_junk_{i}" for i in range(3)]
    for coll in coll_names:
        vecdb.create_collection(collection_name=coll)
    n_colls = len(
        [c for c in vecdb.list_collections(empty=True) if c.startswith("test_junk")]
    )
    n_dels = vecdb.clear_all_collections(really=True, prefix="test_junk")
    assert n_colls == n_dels
