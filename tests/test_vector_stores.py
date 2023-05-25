from llmagent.vector_store.qdrantdb import QdrantDBConfig, QdrantDB
from llmagent.vector_store.chromadb import ChromaDBConfig, ChromaDB
from llmagent.vector_store.base import VectorStore
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)
from llmagent.embedding_models.base import EmbeddingModelsConfig
from llmagent.mytypes import Document
from llmagent.utils.system import rmdir
from dotenv import load_dotenv
from typing import Union
import os
import pytest

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
    api_key=api_key,
)

sentence_cfg = SentenceTransformerEmbeddingsConfig(
    model_type="sentence-transformer",
)


def generate_vecdbs(embed_cfg: EmbeddingModelsConfig) -> VectorStore:
    qd_dir = ".qdrant-" + embed_cfg.model_type
    rmdir(qd_dir)
    qd_cfg = QdrantDBConfig(
        type="qdrant",
        cloud=False,
        collection_name="test-" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    qd_cfg_cloud = QdrantDBConfig(
        type="qdrant",
        cloud=True,
        collection_name="test-" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    cd_dir = ".chroma-" + embed_cfg.model_type
    rmdir(cd_dir)
    cd_cfg = ChromaDBConfig(
        type="chroma",
        collection_name="test-" + embed_cfg.model_type,
        storage_path=cd_dir,
        embedding=embed_cfg,
    )

    qd = QdrantDB(qd_cfg)
    qd_cloud = QdrantDB(qd_cfg_cloud)
    cd = ChromaDB(cd_cfg)

    return [qd, qd_cloud, cd]


@pytest.mark.parametrize(
    "vecdb", generate_vecdbs(openai_cfg) + generate_vecdbs(sentence_cfg)
)
def test_vector_stores(vecdb: Union[ChromaDB, QdrantDB]):
    docs = [
        Document(content="hello", metadata={"id": 1}),
        Document(content="world", metadata={"id": 2}),
        Document(content="hi there", metadata={"id": 2}),
    ]
    vecdb.add_documents(docs)
    docs_and_scores = vecdb.similar_texts_with_scores("hello", k=2)
    assert set([docs_and_scores[0][0].content, docs_and_scores[1][0].content]) == set(
        ["hello", "hi there"]
    )
    if vecdb.config.cloud:
        vecdb.delete_collection(collection_name=vecdb.config.collection_name)
    else:
        rmdir(vecdb.config.storage_path)
