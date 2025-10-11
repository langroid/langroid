import pytest

from langroid.agent.batch import run_batch_tasks
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.qdrantdb import QdrantDBConfig


@pytest.fixture(scope="function")
def local_qdrant_config(monkeypatch) -> QdrantDBConfig:
    monkeypatch.setenv("QDRANT_API_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "local-dev-key")
    return QdrantDBConfig(
        cloud=True,
        collection_name="pytest-concurrent-doc-chat",
        embedding=SentenceTransformerEmbeddingsConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        replace_collection=True,
    )


def test_doc_chat_concurrent_local_qdrant(local_qdrant_config):
    cfg = DocChatAgentConfig(
        name="pytest-agent",
        vecdb=local_qdrant_config,
        retrieve_only=True,
        use_bm25_search=False,
        use_fuzzy_match=False,
        cross_encoder_reranking_model="",
        use_reciprocal_rank_fusion=False,
        relevance_extractor_config=None,
    )
    agent = DocChatAgent(cfg)
    agent.llm = None
    docs = [
        Document(
            content=(
                "The Library is composed of hexagonal galleries filled with books. "
                "Each gallery stores countless volumes with varied letter combinations."
            ),
            metadata=DocMetaData(source="test-doc"),
        )
    ]
    agent.ingest_docs(docs)
    agent.vecdb.config.replace_collection = False

    queries = [
        "What is the structure of the Library described in the story?",
        "What do the books in the Library contain?",
        "What is the significance of the hexagonal galleries?",
    ]

    def retrieve(agent_obj, question: str) -> str:
        chunks = agent_obj.get_relevant_chunks(question, [])
        return "\n".join(chunk.content for chunk in chunks) if chunks else "DO-NOT-KNOW"

    results = run_batch_tasks(
        Task(agent, interactive=False, single_round=True),
        queries,
        sequential=False,
        turns=1,
        output_map=lambda x: x,
    )

    assert all(res is not None and res.content != "DO-NOT-KNOW" for res in results)
