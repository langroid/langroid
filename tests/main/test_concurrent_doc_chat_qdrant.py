import pytest

from langroid.agent.batch import run_batch_tasks
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.embedding_models.models import SentenceTransformerEmbeddingsConfig
from langroid.language_models.mock_lm import MockLM, MockLMConfig
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
    mock_llm_cfg = MockLMConfig(default_response="Mock response")
    agent.config.llm = mock_llm_cfg
    agent.llm = MockLM(mock_llm_cfg)
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
    # Force regression scenario: remove the backing collection so clones must rely
    # on in-memory chunk caches (buggy baseline returns DO-NOT-KNOW here).
    if agent.vecdb is not None and agent.vecdb.config.collection_name is not None:
        agent.vecdb.delete_collection(agent.vecdb.config.collection_name)
        agent.vecdb.config.replace_collection = False

    queries = [
        "What is the structure of the Library described in the story?",
        "What do the books in the Library contain?",
        "What is the significance of the hexagonal galleries?",
    ]

    assert len(agent.chunked_docs) == len(docs)

    clone = agent.clone(1)
    assert len(clone.chunked_docs) == len(agent.chunked_docs)

    results = run_batch_tasks(
        Task(agent, interactive=False, single_round=True),
        queries,
        sequential=False,
        turns=1,
        output_map=lambda x: x,
    )

    assert all(res is not None and "DO-NOT-KNOW" not in res.content for res in results)
