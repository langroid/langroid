from types import SimpleNamespace
from typing import List

import pytest
from dotenv import load_dotenv

from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
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

stored_docs = [
    Document(content=d, metadata=DocMetaData(id=i))
    for i, d in enumerate(vars(phrases).values())
]


@pytest.fixture(scope="module")
def vecdb(request) -> VectorStore:
    if request.param == "qdrant_local":
        qd_dir = ".qdrant/data/" + embed_cfg.model_type
        rmdir(qd_dir)
        qd_cfg = QdrantDBConfig(
            type="qdrant",
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
            type="qdrant",
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
            type="chroma",
            collection_name="test-" + embed_cfg.model_type,
            storage_path=cd_dir,
            embedding=embed_cfg,
        )
        cd = ChromaDB(cd_cfg)
        cd.add_documents(stored_docs)
        yield cd
        rmdir(cd_dir)
        return


@pytest.mark.parametrize(
    "query,results",
    [
        ("which city is Belgium's capital?", [phrases.BELGIUM]),
        ("capital of France", [phrases.FRANCE]),
        ("hello", [phrases.HELLO]),
        ("hi there", [phrases.HI_THERE]),
        ("men and women over 40", [phrases.OVER_40]),
        ("people aged less than 40", [phrases.UNDER_40]),
        ("Canadian residents", [phrases.CANADA]),
        ("people outside Canada", [phrases.NOT_CANADA]),
    ],
)
@pytest.mark.parametrize(
    "vecdb", ["qdrant_local", "qdrant_cloud", "chroma"], indirect=True
)
def test_vector_stores(vecdb, query: str, results: List[str]):
    docs_and_scores = vecdb.similar_texts_with_scores(query, k=len(vars(phrases)))
    # first doc should be best match
    if isinstance(vecdb, ChromaDB):
        # scores are (apparently) l2 distances (docs unclear), so low means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score < 0.2]
    else:
        # scores are cosine similarities, so high means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score > 0.8]
    assert set(results).issubset(set(matching_docs))
