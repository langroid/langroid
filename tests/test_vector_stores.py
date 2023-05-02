from llmagent.vector_store.qdrantdb import QdrantDBConfig, QdrantDB
from llmagent.vector_store.chromadb import ChromaDBConfig, ChromaDB
from llmagent.vector_store.base import VectorStore
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)
from llmagent.embedding_models.base import EmbeddingModelsConfig
from llmagent.mytypes import Document
from dotenv import load_dotenv
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
    qd_cfg = QdrantDBConfig(
        type="qdrant",
        collection_name="test-" + embed_cfg.model_type,
        storage_path=".qdrant/testdata-" + embed_cfg.model_type,
        embedding=embed_cfg,
    )

    cd_cfg = ChromaDBConfig(
        type="chroma",
        collection_name="test-" + embed_cfg.model_type,
        storage_path=".chroma/testdata-" + embed_cfg.model_type,
        embedding=embed_cfg,
    )

    qd = QdrantDB(qd_cfg)
    cd = ChromaDB(cd_cfg)

    return [qd, cd]


@pytest.mark.parametrize(
    "vecdb", generate_vecdbs(openai_cfg) + generate_vecdbs(sentence_cfg)
)
def test_vector_stores(vecdb):
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
