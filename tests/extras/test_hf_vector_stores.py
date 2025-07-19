"""
Test vector stores using HuggingFace embeddings.
This depends on sentence-transformers being installed:
 uv sync --dev --extra hf-embeddings
"""

from typing import Union

import pytest

from langroid.embedding_models.base import EmbeddingModelsConfig
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig
from langroid.embedding_models.remote_embeds import RemoteEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

sentence_cfg = SentenceTransformerEmbeddingsConfig(
    model_type="sentence-transformer",
)
remote_cfg = RemoteEmbeddingsConfig()


def generate_vecdbs(embed_cfg: EmbeddingModelsConfig) -> list[VectorStore]:
    qd_dir = ".qdrant-" + embed_cfg.model_type
    rmdir(qd_dir)
    qd_cfg = QdrantDBConfig(
        cloud=False,
        collection_name="test-" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    qd_cfg_cloud = QdrantDBConfig(
        cloud=True,
        collection_name="test-" + embed_cfg.model_type,
        storage_path=qd_dir,
        embedding=embed_cfg,
    )

    cd_dir = ".chroma-" + embed_cfg.model_type
    rmdir(cd_dir)
    cd_cfg = ChromaDBConfig(
        collection_name="test-" + embed_cfg.model_type,
        storage_path=cd_dir,
        embedding=embed_cfg,
    )

    qd = QdrantDB(qd_cfg)
    qd_cloud = QdrantDB(qd_cfg_cloud)
    cd = ChromaDB(cd_cfg)

    return [qd, qd_cloud, cd]


@pytest.mark.parametrize(
    "vecdb", generate_vecdbs(sentence_cfg) + generate_vecdbs(remote_cfg)
)
def test_vector_stores(vecdb: Union[ChromaDB, QdrantDB]):
    docs = [
        Document(content="hello", metadata=DocMetaData(id="1")),
        Document(content="world", metadata=DocMetaData(id="2")),
        Document(content="hi there", metadata=DocMetaData(id="3")),
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
