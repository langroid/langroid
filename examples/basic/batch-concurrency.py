"""Offline rolling-concurrency demo and reproducible equal-capacity benchmark.

Run the MockLM demo:
    uv run python examples/basic/batch-concurrency.py

Save a benchmark (no model calls):
    uv run python examples/basic/batch-concurrency.py --benchmark --output-dir results
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import platform
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

from langroid.agent.batch import (
    ExceptionHandling,
    run_batch_tasks,
    run_batched_tasks,
)
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global


def demo() -> None:
    """Run real Task clones with a local MockLM and a rolling limit of two."""

    async def respond(message: str) -> str:
        await asyncio.sleep(0.1 if message == "slow" else 0.01)
        return f"Processed {message}"

    agent = ChatAgent(
        ChatAgentConfig(
            name="OfflineWorker",
            vecdb=None,
            llm=MockLMConfig(response_fn_async=respond),
        )
    )
    task = Task(agent, interactive=False, done_if_response=[Entity.LLM])
    results = run_batch_tasks(
        task,
        ["slow", "fast-1", "fast-2"],
        sequential=False,
        max_concurrency=2,
        output_map=lambda result: None if result is None else result.content,
    )
    print(results)


def measure(workload: str, mode: str, n: int, capacity: int) -> dict[str, Any]:
    """Measure one public call, retaining only its own task references."""
    durations = [
        0.2 if workload == "long_tail" and index % 10 == 0 else 0.02
        for index in range(n)
    ]
    started = [0.0] * n
    completed = [0.0] * n
    owned: set[asyncio.Task[Any]] = set()
    active = 0
    peak_active = 0

    async def work(value: str | ChatDocument, index: int) -> int:
        nonlocal active, peak_active
        current = asyncio.current_task()
        assert current is not None
        owned.add(current)
        started[index] = time.perf_counter() - submitted
        active += 1
        peak_active = max(active, peak_active)
        try:
            await asyncio.sleep(durations[index])
            return index
        finally:
            completed[index] = time.perf_counter() - submitted
            active -= 1

    inputs = [str(index) for index in range(n)]
    submitted = time.perf_counter()
    results = run_batched_tasks(
        inputs=inputs,
        do_task=work,
        batch_size=capacity if mode == "fixed" else None,
        max_concurrency=capacity if mode == "rolling" else None,
        sequential=False,
        stop_on_first_result=False,
        handle_exceptions=ExceptionHandling.RAISE,
        output_map=lambda value: value,
        message_template="Offline concurrency benchmark",
    )
    elapsed = time.perf_counter() - submitted
    assert results == list(range(n))
    assert peak_active <= capacity and active == 0
    residual = sum(not task.done() for task in owned)
    assert residual == 0
    p95_index = math.ceil(n * 0.95) - 1
    return dict(
        workload=workload,
        mode=mode,
        n=n,
        capacity=capacity,
        elapsed_seconds=elapsed,
        throughput_per_second=n / elapsed,
        queue_p95_seconds=sorted(started)[p95_index],
        end_to_end_p95_seconds=sorted(completed)[p95_index],
        peak_active=peak_active,
        residual_tasks=residual,
        items=[
            dict(
                index=i,
                duration_seconds=durations[i],
                queue_seconds=started[i],
                end_to_end_seconds=completed[i],
            )
            for i in range(n)
        ],
    )


def benchmark(output_dir: Path) -> None:
    """Alternate modes after warmup; measure Python memory separately."""
    n, capacity, repetitions = 100, 10, 5
    runs: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for workload in ("uniform", "long_tail"):
        for mode in ("fixed", "rolling"):
            measure(workload, mode, n, capacity)
        for repetition in range(repetitions):
            modes = (
                ("fixed", "rolling") if repetition % 2 == 0 else ("rolling", "fixed")
            )
            for mode in modes:
                row = measure(workload, mode, n, capacity)
                row["repetition"] = repetition + 1
                runs.append(row)
        for mode in ("fixed", "rolling"):
            tracemalloc.start()
            try:
                measure(workload, mode, n, capacity)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            memory.append(dict(workload=workload, mode=mode, peak_python_bytes=peak))

    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    metadata = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        baseline_sha=git("rev-parse", "HEAD"),
        git_status=git("status", "--short"),
        batch_source_sha256=hashlib.sha256(
            (repo / "langroid/agent/batch.py").read_bytes()
        ).hexdigest(),
        n=n,
        capacity=capacity,
        repetitions=repetitions,
        workload=(
            "Deterministic: uniform 20ms; long-tail every tenth 200ms, " "others 20ms"
        ),
        submission="All items submitted at public-call entry, before scheduling",
        memory=(
            "Separate tracemalloc run; Python allocations including "
            "instrumentation, not RSS"
        ),
        logging=(
            "logging disabled; quiet=True, progress=False, cache=False; "
            "no external model or service"
        ),
        statistic=(
            "P95 nearest rank; aggregate speedup is ratio of median elapsed times"
        ),
    )
    summaries = []
    for workload in ("uniform", "long_tail"):
        fixed, rolling = [
            statistics.median(
                row["elapsed_seconds"]
                for row in runs
                if row["workload"] == workload and row["mode"] == mode
            )
            for mode in ("fixed", "rolling")
        ]
        summaries.append(
            dict(
                workload=workload,
                fixed_median_seconds=fixed,
                rolling_median_seconds=rolling,
                speedup=fixed / rolling,
                elapsed_reduction_fraction=(fixed - rolling) / fixed,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(
            dict(metadata=metadata, runs=runs, memory=memory, summary=summaries),
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[key for key in runs[0] if key != "items"]
        )
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "items"} for row in runs
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    set_global(
        Settings(quiet=True, progress=False, cache=False, cache_type="fakeredis")
    )
    if args.benchmark:
        if args.output_dir is None:
            parser.error("--benchmark requires --output-dir")
        logging.disable(logging.CRITICAL)
        benchmark(args.output_dir)
    else:
        demo()
