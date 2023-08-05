from typing import List, Union

import pytest
from dotenv import load_dotenv

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

load_dotenv()
openai_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)


def generate_vecdbs(embed_cfg: EmbeddingModelsConfig) -> VectorStore:
    qd_dir = ".qdrant/cloud/" + embed_cfg.model_type
    qd_cfg_cloud = QdrantDBConfig(
        type="qdrant",
        cloud=True,
        collection_name="test-" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    cd_dir = ".chroma/" + embed_cfg.model_type
    rmdir(cd_dir)
    cd_cfg = ChromaDBConfig(
        type="chroma",
        collection_name="test-" + embed_cfg.model_type,
        storage_path=cd_dir,
        embedding=embed_cfg,
    )

    qd_cloud = QdrantDB(qd_cfg_cloud)
    cd = ChromaDB(cd_cfg)

    return [qd_cloud, cd]


@pytest.mark.parametrize(
    "query,results",
    [
        ("what is the capital of Belgium?", ["which city is Belgium's capital?"]),
        ("hello", ["hello"]),
        ("men and women over 40", ["people over 40"]),
        ("people under 40", ["people under 40"]),
        ("Canadian residents", ["people living in Canada"]),
        ("People living outside Canada", ["people not living in Canada"]),
    ],
)
@pytest.mark.parametrize("vecdb", generate_vecdbs(openai_cfg))
def test_vector_stores(
    vecdb: Union[ChromaDB, QdrantDB], query: str, results: List[str]
):
    docs = [
        Document(content="hello", metadata=DocMetaData(id=1)),
        Document(content="hi there", metadata=DocMetaData(id=2)),
        Document(content="people living in Canada", metadata=DocMetaData(id=3)),
        Document(content="people not living in Canada", metadata=DocMetaData(id=4)),
        Document(content="people over 40", metadata=DocMetaData(id=5)),
        Document(content="people under 40", metadata=DocMetaData(id=6)),
        Document(content="what is the capital of France?", metadata=DocMetaData(id=7)),
        Document(
            content="which city is Belgium's capital?", metadata=DocMetaData(id=8)
        ),
    ]
    vecdb.add_documents(docs)
    docs_and_scores = vecdb.similar_texts_with_scores(query, k=8)
    # first doc should be best match
    if isinstance(vecdb, ChromaDB):
        # scores are (apparently) l2 distances (docs unclear), so low means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score < 0.2]
    else:
        # scores are cosine similarities, so high means close
        matching_docs = [doc.content for doc, score in docs_and_scores if score > 0.8]
    assert set(results).issubset(set(matching_docs))
    if vecdb.config.cloud:
        vecdb.delete_collection(collection_name=vecdb.config.collection_name)
    else:
        rmdir(vecdb.config.storage_path)
