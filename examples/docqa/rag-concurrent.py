"""
Concurrent RAG example using DocChatAgent with custom asyncio harness

This example demonstrates running multiple DocChat queries concurrently
with detailed live logging that shows every task starting and finishing
in real time (no waiting for gather() to return), making concurrency
easy to verify at a glance.

IMPORTANT: The --sequential flag runs tasks in a TRUE sequential loop
(not asyncio's sequential mode), providing a baseline for comparison.

Usage:

# Run concurrently with asyncio (default)
python3 examples/docqa/rag-concurrent.py

# Run in TRUE sequential mode (simple loop) for baseline comparison
python3 examples/docqa/rag-concurrent.py --sequential

# With specific model
python3 examples/docqa/rag-concurrent.py -m ollama/mistral:7b-instruct-v0.2-q8_0

# Use local SentenceTransformer embeddings with Docker Qdrant on localhost:6333
python3 examples/docqa/rag-concurrent.py --local-embeddings

# Compare both modes to measure concurrency speedup
python3 examples/docqa/rag-concurrent.py --sequential  # Baseline
python3 examples/docqa/rag-concurrent.py  # Should be faster if truly concurrent

# Use Langroid's built-in run_batch_tasks harness instead of the custom one
python3 examples/docqa/rag-concurrent.py --use-builtin-batch

# Show only concurrency logs (suppress long answers) and filter to START/WORKER lines
python3 examples/docqa/rag-concurrent.py --num-questions=3 --log-only \\
  | rg "Q[0-9]{2} (START|WORKER|COMPLETE)"

The logs show:
- Timestamps (HH:MM:SS.mmm) for each task start/complete
- Thread IDs to verify parallel execution
- Question numbers for tracking

Expected patterns:
- SEQUENTIAL: START->COMPLETE->START->COMPLETE (one at a time)
- CONCURRENT: Multiple STARTs with close timestamps before any COMPLETEs

If concurrent mode shows START->COMPLETE pattern, there's a bottleneck
(e.g., shared vecdb client causing serialization).

See here for more on how to set up a local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import asyncio
import os
import time
import threading
from datetime import datetime
from contextvars import ContextVar
from typing import Dict

import fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.batch import run_batch_tasks, run_batch_task_gen
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Thread-safe logging with timestamps
log_lock = threading.Lock()
CURRENT_QUESTION: ContextVar[int | None] = ContextVar("CURRENT_QUESTION", default=None)
EVENT_HISTORY: list[str] = []
QUESTION_TO_INDEX: Dict[str, int] = {}


def log_event(event_type: str, question_num: int, message: str = ""):
    """Thread-safe logging with precise timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    thread_id = threading.get_ident() % 10000  # Short thread ID
    line = f"[{timestamp}] [{thread_id:04d}] Q{question_num:02d} {event_type:12s} {message}"
    EVENT_HISTORY.append(line)
    with log_lock:
        print(line)


# 10 questions about Borges' "The Library of Babel"
ALL_QUESTIONS = [
    "What is the structure of the Library described in the story?",
    "What do the books in the Library contain?",
    "What is the significance of the hexagonal galleries?",
    "How many books are estimated to exist in the Library?",
    "What is the narrator's theory about the origin of the Library?",
    "How does the story describe the contents of most books?",
    "What happens to librarians who search for meaningful books?",
    "What is the emotional impact of the infinite Library on the librarians?",
    "What philosophical themes does the story explore?",
    "What is the relationship between infinity and meaning in the story?",
]


class LoggingDocChatAgent(DocChatAgent):
    """DocChatAgent that reports worker-thread execution for visibility."""

    def answer_from_docs(self, query: str):
        q_num = CURRENT_QUESTION.get()
        if q_num is None:
            q_num = QUESTION_TO_INDEX.get(query)
        if q_num is not None:
            log_event(
                "WORKER_START", q_num, f"Vec/LLM on T{threading.get_ident()%10000:04d}"
            )
            start = time.time()
        result = super().answer_from_docs(query)
        if q_num is not None:
            elapsed = time.time() - start
            log_event(
                "WORKER_DONE",
                q_num,
                f"{elapsed:.2f}s on T{threading.get_ident()%10000:04d}",
            )
        return result


def app(
    m: str = "",
    sequential: bool = False,
    num_questions: int = 10,
    log_only: bool = False,
    use_builtin_batch: bool = False,
    local_embeddings: bool = False,
):
    """
    Run DocChat queries on Library of Babel story.

    Args:
        m: Model name (default: GPT-4o)
        sequential: If True, run truly sequentially (simple loop);
                   if False, run with asyncio concurrency (default: False)
        num_questions: Number of questions to run (max 10)
        log_only: Suppress verbose answers and print a concise log summary
        use_builtin_batch: Use Langroid's run_batch_tasks instead of the custom harness
    """
    num_questions = max(1, min(num_questions, len(ALL_QUESTIONS)))
    questions = ALL_QUESTIONS[:num_questions]
    QUESTION_TO_INDEX.clear()
    QUESTION_TO_INDEX.update({q: i + 1 for i, q in enumerate(questions)})
    EVENT_HISTORY.clear()
    mode = "TRULY SEQUENTIAL (simple loop)" if sequential else "CONCURRENT (asyncio)"
    print(f"\n{'='*80}")
    print(f"Running in {mode} mode")
    print(f"{'='*80}\n")

    # Create the llm config object
    llm_config = lm.OpenAIGPTConfig(
        chat_model=m or lm.OpenAIChatModel.GPT4o,
        chat_context_length=32_000,
        max_output_tokens=300,
        temperature=0.2,
        stream=False,  # Disable streaming for batch processing
        timeout=45,
    )

    # Configure DocChatAgent with Library of Babel story
    vecdb_config = None
    if local_embeddings:
        try:
            from langroid.embedding_models.models import (
                SentenceTransformerEmbeddingsConfig,
            )
            from langroid.vector_store.qdrantdb import QdrantDBConfig
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformer embeddings require the hf-embeddings extras"
            ) from exc

        os.environ.setdefault("QDRANT_API_URL", "http://localhost:6333")
        os.environ.setdefault("QDRANT_API_KEY", "local-dev-key")

        sentence_cfg = SentenceTransformerEmbeddingsConfig(
            model_type="sentence-transformer",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        vecdb_config = QdrantDBConfig(
            cloud=True,
            collection_name="doc-chat-local-embeddings",
            replace_collection=True,
            embedding=sentence_cfg,
        )

    config = DocChatAgentConfig(
        name="RagAgent",
        llm=llm_config,
        relevance_extractor_config=None,
        vecdb=vecdb_config,
    )

    # Create agent and ingest the document
    agent = LoggingDocChatAgent(config)
    url = "https://xpressenglish.com/our-stories/library-of-babel/"
    print(f"\nIngesting document: {url}")
    agent.ingest_doc_paths([url])
    print("Document ingested successfully.\n")
    if local_embeddings and agent.vecdb is not None:
        agent.vecdb.config.replace_collection = False

    # Create a single task that will be cloned for each question
    print(f"Creating task for concurrent execution of {len(questions)} queries...\n")

    task = lr.Task(
        agent,
        interactive=False,
        single_round=True,
    )

    # Run tasks and measure time
    print("\n" + "=" * 80)
    print("EXECUTION LOG (with timestamps and thread IDs)")
    print("=" * 80 + "\n")
    start_time = time.time()

    if sequential:
        # TRUE SEQUENTIAL: Simple loop, no async
        results = []
        for i, question in enumerate(questions, 1):
            log_event(
                "START", i, question[:50] + "..." if len(question) > 50 else question
            )  # noqa: E501
            token = CURRENT_QUESTION.set(i)
            try:
                result = task.run(question, turns=1)
            finally:
                CURRENT_QUESTION.reset(token)
            log_event(
                "COMPLETE",
                i,
                f"Got response ({len(str(result.content if result else ''))} chars)",
            )  # noqa: E501
            final = (
                result.content
                if result and hasattr(result, "content")
                else str(result) if result else ""
            )  # noqa: E501
            results.append(final)
    else:
        if use_builtin_batch:

            def input_map(question: str) -> str:
                q_num = QUESTION_TO_INDEX[question]
                log_event(
                    "START",
                    q_num,
                    question[:50] + "..." if len(question) > 50 else question,
                )
                return question

            # run_batch_task_gen allows handle_exceptions to crash on errors
            def gen_task(i: int) -> lr.Task:
                return task.clone(i)

            raw_results_gen = run_batch_task_gen(
                gen_task=gen_task,
                items=questions,
                input_map=input_map,
                sequential=False,
                turns=1,
                handle_exceptions=False,  # Crash on errors to see what's failing
            )
            raw_results = list(raw_results_gen)
            results = []
            for i, result in enumerate(raw_results, 1):
                if result is None:
                    length = 0
                    results.append("")
                elif hasattr(result, "content"):
                    length = len(result.content)
                    results.append(result)
                else:
                    text = str(result)
                    length = len(text)
                    results.append(text)
                log_event("COMPLETE", i, f"Got response ({length} chars)")
        else:
            # CONCURRENT: Custom asyncio runner using task clones and as_completed

            async def run_question(clone_idx: int, question: str, base_task: lr.Task):
                """Launch a clone of the base task and report progress live."""
                q_num = clone_idx + 1
                log_event(
                    "START",
                    q_num,
                    question[:50] + "..." if len(question) > 50 else question,
                )
                token = CURRENT_QUESTION.set(q_num)
                task_clone = base_task.clone(clone_idx)
                try:
                    result = await task_clone.run_async(question, turns=1)
                finally:
                    CURRENT_QUESTION.reset(token)
                if result is None:
                    length = 0
                elif hasattr(result, "content"):
                    length = len(result.content)
                else:
                    length = len(str(result))
                log_event("COMPLETE", q_num, f"Got response ({length} chars)")
                return q_num, result

            async def run_all_concurrent():
                coros = [
                    run_question(idx, question, task)
                    for idx, question in enumerate(questions)
                ]
                results_ordered = [None] * len(questions)
                for coro in asyncio.as_completed(coros):
                    q_num, result = await coro
                    results_ordered[q_num - 1] = result
                return results_ordered

            results = asyncio.run(run_all_concurrent())

    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed {len(questions)} queries in {elapsed_time:.2f} seconds")
    print(f"Average time per query: {elapsed_time/len(questions):.2f} seconds")
    print(f"{'='*80}\n")

    if log_only:
        print("\nLOG SUMMARY (captured START/WORKER/COMPLETE events)")
        print("-" * 80)
        for line in EVENT_HISTORY:
            print(line)
        print("-" * 80)
    else:
        print("\nINTERPRETING THE LOGS:")
        print("-" * 80)
        if sequential:
            print("SEQUENTIAL MODE: Tasks run one at a time in a simple loop")
            print("You should see: START->COMPLETE->START->COMPLETE pattern")
            print("This is the baseline for comparison.")
        else:
            print("CONCURRENT MODE: Tasks should run in parallel with asyncio")
            print(
                "Expected: Multiple 'START' events with close timestamps BEFORE any 'COMPLETE'"
            )  # noqa: E501
            print("If you see START->COMPLETE->START->COMPLETE instead,")
            print(
                "then there's a bottleneck preventing concurrency (e.g., shared vecdb)"
            )
            print("\nThread IDs: Different IDs = parallel execution")
            print("Timestamps: Overlapping windows = true concurrency")
        print("-" * 80 + "\n")

        # Display results
        for i, (question, result) in enumerate(zip(questions, results), 1):
            print(f"\n{'='*80}")
            print(f"Q{i}: {question}")
            print(f"{'-'*80}")
            if result is not None:
                answer = result.content if hasattr(result, "content") else str(result)
            else:
                answer = "No response"
            print(f"A{i}: {answer}")
            print(f"{'='*80}")

    QUESTION_TO_INDEX.clear()
    return results


if __name__ == "__main__":
    fire.Fire(app)
