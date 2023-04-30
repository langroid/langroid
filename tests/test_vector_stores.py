from llmagent.vector_store.qdrantdb import QdrantDBConfig, QdrantDB
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)
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

@pytest.mark.parametrize("embed_cfg", [
    # pytest.param(openai_cfg, id="openai"), disable for cost reasons
    pytest.param(sentence_cfg, id="sentencetransformer"),
])
def test_vector_stores(embed_cfg):
    qd_cfg = QdrantDBConfig(
        type = "qdrant",
        collection_name = "test",
        storage_path = ".qdrant/testdata",
        embedding = embed_cfg,
    )

    qd = QdrantDB(qd_cfg)

    docs = [
        Document(content="hello", metadata={"id": 1}),
        Document(content="world", metadata={"id": 2}),
        Document(content="hi there", metadata={"id": 2}),
    ]
    qd.add_documents(docs)
    docs_and_scores = qd.similar_texts_with_scores("hello", k=2)
    assert ( set([docs_and_scores[0][0].content, docs_and_scores[1][0].content]) ==
             set(["hello", "hi there"]))




