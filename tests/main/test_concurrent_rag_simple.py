"""
Simplified standalone test to reproduce concurrent RAG cross-encoder race condition.

This test uses the Borges "Library of Babel" story with auto-ingestion and 2 parallel
tasks to trigger the cross-encoder race condition. Using n_similar_chunks=10 increases
time spent in cross-encoder reranking, increasing collision probability.

═══════════════════════════════════════════════════════════════════════════════
DOCKER QDRANT SETUP (REQUIRED)
═══════════════════════════════════════════════════════════════════════════════

The bug is SPECIFIC to concurrent access to Docker Qdrant.

Step 1: Create directory and start Docker Qdrant with volume mount
    mkdir -p qdrantdb_docker
    docker run -d \
      --name test-qdrant \
      -p 6333:6333 -p 6334:6334 \
      -v $(pwd)/qdrantdb_docker:/qdrant/storage \
      qdrant/qdrant

Step 2: Run this test - it will auto-ingest the Borges story
    pytest -xvs tests/test_concurrent_rag_simple.py

═══════════════════════════════════════════════════════════════════════════════

Error reproduced:
    NotImplementedError: Cannot copy out of meta tensor; no data!

Root cause:
    Multiple threads in run_batch_task_gen() simultaneously call
    rerank_with_cross_encoder(), which tries to move the shared
    cross-encoder model to a device, causing a PyTorch race condition.

GPU/MPS validation:
    pytest tests/main/test_concurrent_rag_simple.py -k cross_encoder -x \
        --cross-encoder-device=mps
"""

import os
from typing import Optional

import pytest

import langroid as lr
import langroid.language_models as lm
from langroid.agent.batch import run_batch_task_gen
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.utils.configuration import settings

COLLECTION_NAME = "borges-babel-test"
BORGES_URL = "https://xpressenglish.com/our-stories/library-of-babel/"

settings.cache = False


DEVICE_OVERRIDE: Optional[str] = None


def pytest_addoption(parser):
    parser.addoption(
        "--cross-encoder-device",
        action="store",
        default=None,
        help=(
            "Device string for cross-encoder reranker (e.g. 'cpu', 'cuda', 'mps'). "
            "Overrides TEST_CROSS_ENCODER_DEVICE env var."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def _set_device_override(request):
    global DEVICE_OVERRIDE
    cli_device = request.config.getoption("cross_encoder_device")
    env_device = os.getenv("TEST_CROSS_ENCODER_DEVICE")
    DEVICE_OVERRIDE = cli_device or env_device


def setup_rag_agent() -> lr.Task:
    """
    Create a DocChatAgent with the Borges story ingested.

    Returns:
        Langroid Task with DocChatAgent configured for RAG
    """
    llm_config = lm.MockLMConfig(default_response="ok")

    embed_cfg = lr.embedding_models.SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="BAAI/bge-large-en-v1.5",
    )

    config = DocChatAgentConfig(
        name="DocAgent",
        llm=llm_config,
        n_query_rephrases=0,
        assistant_mode=True,
        hypothetical_answer=False,
        n_neighbor_chunks=1,
        n_similar_chunks=10,  # Retrieve 10 chunks - increases cross-encoder workload
        n_relevant_chunks=10,  # Keep all 10 chunks after reranking
        # Enable cross-encoder reranking
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_device=DEVICE_OVERRIDE,
        relevance_extractor_config=None,  # Disable LLM-based relevance extraction
        parsing=ParsingConfig(
            splitter=Splitter.TOKENS,
            chunk_size=200,
            overlap=50,
        ),
        vecdb=lr.vector_store.QdrantDBConfig(
            cloud=False,
            docker=True,
            collection_name=COLLECTION_NAME,
            embedding=embed_cfg,
            host="localhost",
            port="6333",
            replace_collection=True,  # Start fresh each time
        ),
    )

    agent = DocChatAgent(config)
    agent.vecdb.set_collection(COLLECTION_NAME)

    # Ingest the Borges story
    print(f"\nIngesting document: {BORGES_URL}")
    agent.ingest_doc_paths([BORGES_URL])
    print("Ingestion complete!")

    task = lr.Task(agent, interactive=False, single_round=True)
    return task


def create_rag_agent() -> lr.Task:
    """
    Create a DocChatAgent that connects to existing Borges collection.

    Returns:
        Langroid Task with DocChatAgent
    """
    llm_config = lm.MockLMConfig(default_response="ok")

    embed_cfg = lr.embedding_models.SentenceTransformerEmbeddingsConfig(
        model_type="sentence-transformer",
        model_name="BAAI/bge-large-en-v1.5",
    )

    config = DocChatAgentConfig(
        name="DocAgent",
        llm=llm_config,
        n_query_rephrases=0,
        assistant_mode=True,
        hypothetical_answer=False,
        n_neighbor_chunks=1,
        n_similar_chunks=10,  # Retrieve 10 chunks - increases cross-encoder workload
        n_relevant_chunks=10,  # Keep all 10 chunks after reranking
        # Enable cross-encoder reranking
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_device=DEVICE_OVERRIDE,
        relevance_extractor_config=None,  # Disable LLM-based relevance extraction
        parsing=ParsingConfig(
            splitter=Splitter.TOKENS,
            chunk_size=200,
            overlap=50,
        ),
        vecdb=lr.vector_store.QdrantDBConfig(
            cloud=False,
            docker=True,
            collection_name=COLLECTION_NAME,
            embedding=embed_cfg,
            host="localhost",
            port="6333",
            replace_collection=False,  # Reuse existing collection
        ),
    )

    agent = DocChatAgent(config)
    agent.vecdb.set_collection(COLLECTION_NAME)

    task = lr.Task(agent, interactive=False, single_round=True)
    return task


def test_concurrent_rag_cross_encoder_race_condition():
    """
    Test that reproduces the cross-encoder race condition bug.

    Two parallel tasks query the same Borges collection to trigger
    concurrent access to the cross-encoder model.
    """
    print("\n" + "=" * 80)
    print("SETUP: Creating first RAG agent and ingesting Borges story...")
    print("=" * 80)

    # First task ingests the document
    setup_rag_agent()

    # Second task reuses the collection
    print("\nCreating more RAG agents (reusing collection)...")
    N = 10  # large to increase chance of race condition

    tasks = [create_rag_agent() for _ in range(N)]

    # Same query for both tasks
    query = "What is the Library of Babel?"

    # Disable streaming on all tasks
    # for task in tasks:
    #     if hasattr(task.agent.config.llm, "stream"):
    #         task.agent.config.llm.stream = False

    # Generator function that returns DIFFERENT tasks based on index
    def gen_task(idx: int) -> lr.Task:
        print(f"[Test] Task {idx+1} running query...")
        return tasks[idx]

    print("\n" + "=" * 80)
    print("TEST: Running queries in PARALLEL (should trigger race condition)...")
    print("=" * 80)

    # Run in PARALLEL - this triggers the bug
    results_list = run_batch_task_gen(
        gen_task=gen_task,
        items=[query] * N,  # Same query for all tasks
        sequential=False,  # PARALLEL execution triggers race condition
        input_map=lambda q: q,
        output_map=lambda result: (
            result.content if result is not None else "DO-NOT-KNOW"
        ),
        handle_exceptions=False,  # Let exceptions propagate
    )

    # If we get here, bug didn't occur
    print(f"\n[Test] Results received: {len(results_list)}")
    for i, result in enumerate(results_list):
        if result:
            print(f"\nTask {i+1} answer: {result[:200]}...")

    assert len(results_list) == len(tasks)
    assert all(result is not None for result in results_list)


def test_sequential_rag_no_race_condition():
    """
    Control test: running queries sequentially should NOT trigger the bug.
    """
    print("\n" + "=" * 80)
    print("CONTROL TEST: Running queries SEQUENTIALLY...")
    print("=" * 80)

    task = setup_rag_agent()

    queries = [
        "What is the Library of Babel?",
        "What do the books in the library contain?",
    ]

    if hasattr(task.agent.config.llm, "stream"):
        task.agent.config.llm.stream = False

    # Run SEQUENTIALLY using list comprehension - should NOT trigger the bug
    results_list = []
    for i, query in enumerate(queries):
        print(f"[Test] Running query {i+1}: {query}")
        result = task.run(query)
        output = result.content if result is not None else "DO-NOT-KNOW"
        results_list.append(output)

    print(f"\n[Test] Sequential results received: {len(results_list)}")
    for i, result in enumerate(results_list):
        if result:
            print(f"\nQuery {i+1}: {queries[i]}")
            print(f"Answer: {result[:200]}...")

    assert len(results_list) == len(queries)
    assert all(result is not None for result in results_list)

    print("\n✅ Control test passed - no race condition in sequential mode")


if __name__ == "__main__":
    print("=" * 80)
    print("TEST 1: Concurrent queries (should trigger cross-encoder race condition)")
    print("=" * 80)
    try:
        test_concurrent_rag_cross_encoder_race_condition()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

    print("\n" + "=" * 80)
    print("TEST 2: Sequential queries (control, should NOT trigger bug)")
    print("=" * 80)
    try:
        test_sequential_rag_no_race_condition()
    except Exception as e:
        print(f"\n❌ Control test failed: {e}")
