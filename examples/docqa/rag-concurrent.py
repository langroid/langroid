"""
Concurrent RAG example using DocChatAgent with batch task mechanism

This example demonstrates running multiple DocChat queries concurrently
using Langroid's batch task functionality, with detailed logging to verify
concurrent execution.

IMPORTANT: The --sequential flag runs tasks in a TRUE sequential loop
(not asyncio's sequential mode), providing a baseline for comparison.

Usage:

# Run concurrently with asyncio (default)
python3 examples/docqa/rag-concurrent.py

# Run in TRUE sequential mode (simple loop) for baseline comparison
python3 examples/docqa/rag-concurrent.py --sequential

# With specific model
python3 examples/docqa/rag-concurrent.py -m ollama/mistral:7b-instruct-v0.2-q8_0

# Compare both modes to measure concurrency speedup
python3 examples/docqa/rag-concurrent.py --sequential  # Baseline
python3 examples/docqa/rag-concurrent.py  # Should be faster if truly concurrent

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

import os
import time
import threading
from datetime import datetime

import fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.batch import run_batch_tasks

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Thread-safe logging with timestamps
log_lock = threading.Lock()


def log_event(event_type: str, question_num: int, message: str = ""):
    """Thread-safe logging with precise timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    thread_id = threading.get_ident() % 10000  # Short thread ID
    with log_lock:
        print(
            f"[{timestamp}] [{thread_id:04d}] Q{question_num:02d} {event_type:10s} {message}"
        )  # noqa: E501


# 10 questions about Borges' "The Library of Babel"
QUESTIONS = [
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


def app(m="", sequential=False):
    """
    Run DocChat queries on Library of Babel story.

    Args:
        m: Model name (default: GPT-4o)
        sequential: If True, run truly sequentially (simple loop);
                   if False, run with asyncio concurrency (default: False)
    """
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
    config = DocChatAgentConfig(
        name="RagAgent",
        llm=llm_config,
        relevance_extractor_config=None,
    )

    # Create agent and ingest the document
    agent = DocChatAgent(config)
    url = "https://xpressenglish.com/our-stories/library-of-babel/"
    print(f"\nIngesting document: {url}")
    agent.ingest_doc_paths([url])
    print("Document ingested successfully.\n")

    # Create a single task that will be cloned for each question
    print(f"Creating task for concurrent execution of {len(QUESTIONS)} queries...\n")

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

    # Track question index for logging
    question_map = {q: i + 1 for i, q in enumerate(QUESTIONS)}

    if sequential:
        # TRUE SEQUENTIAL: Simple loop, no async
        results = []
        for i, question in enumerate(QUESTIONS, 1):
            log_event(
                "START", i, question[:50] + "..." if len(question) > 50 else question
            )  # noqa: E501
            result = task.run(question, turns=1)
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
        # CONCURRENT: Use asyncio batch processing
        def input_map(question):
            """Log when each task starts"""
            q_num = question_map[question]
            log_event(
                "START",
                q_num,
                question[:50] + "..." if len(question) > 50 else question,
            )  # noqa: E501
            return question

        def output_map(result):
            """Log when each task completes"""
            final = result.content if hasattr(result, "content") else str(result)
            return final

        results = run_batch_tasks(
            task=task,
            items=QUESTIONS,
            input_map=input_map,
            output_map=output_map,
            sequential=False,  # Use asyncio.gather for true concurrency
            turns=1,
        )

        # Log completion for each result
        for i, result in enumerate(results, 1):
            if result:
                log_event("COMPLETE", i, f"Got response ({len(str(result))} chars)")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed {len(QUESTIONS)} queries in {elapsed_time:.2f} seconds")
    print(f"Average time per query: {elapsed_time/len(QUESTIONS):.2f} seconds")
    print(f"{'='*80}\n")

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
        print("then there's a bottleneck preventing concurrency (e.g., shared vecdb)")
        print("\nThread IDs: Different IDs = parallel execution")
        print("Timestamps: Overlapping windows = true concurrency")
    print("-" * 80 + "\n")

    # Display results
    for i, (question, result) in enumerate(zip(QUESTIONS, results), 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {question}")
        print(f"{'-'*80}")
        if result is not None:
            answer = result.content if hasattr(result, "content") else str(result)
        else:
            answer = "No response"
        print(f"A{i}: {answer}")
        print(f"{'='*80}")

    return results


if __name__ == "__main__":
    fire.Fire(app)
