"""Regression-test for the shared-FD RichFileLogger implementation."""

from __future__ import annotations

import resource  # noqa: E402  pylint: disable=wrong-import-position
import sys
import threading
from pathlib import Path
from typing import List

import pytest

from langroid.utils.logging import RichFileLogger


def _create_many(log_path: str, n: int = 20) -> List[RichFileLogger]:
    """Return `n` RichFileLogger instances that target the same file."""
    return [RichFileLogger(log_path, append=True, color=False) for _ in range(n)]


@pytest.mark.parametrize("n_loggers", [1, 5, 20])
def test_single_fd_reused(tmp_path: Path, n_loggers: int) -> None:
    """
    Ensure that all RichFileLogger instances writing to the same file
    share ONE open file descriptor, and that `.close()` frees it.
    """
    file_path = tmp_path / "shared.log"
    loggers = _create_many(str(file_path), n=n_loggers)

    # All instances must be the very same object (singleton per file)
    first = loggers[0]
    assert all(inst is first for inst in loggers)

    # They must have the exact same underlying fd
    first_fd = first.file.fileno()
    assert all(inst.file.fileno() == first_fd for inst in loggers)

    # Write something; should not raise
    first.log("hi")

    # Closing once must mark the fd closed and remove the instance
    first.close()
    assert first.file.closed
    assert str(file_path) not in RichFileLogger._instances

    # Creating again after close should yield a fresh instance + fd
    fresh = RichFileLogger(str(file_path), append=True, color=False)
    assert fresh is not first
    assert not fresh.file.closed
    fresh.close()


# Linux/mac only – `resource` is not available on Windows.


def _stress_writer(
    start: threading.Event,
    errors: List[BaseException],
    log_path: str,
) -> None:
    """Wait for the `start` event then write to the shared logger many times."""
    start.wait()
    try:
        logger = RichFileLogger(log_path, append=True, color=False)
        for _ in range(50):
            logger.log("hello")
    except BaseException as exc:  # want to catch the OSError 24
        errors.append(exc)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="resource.RLIMIT_NOFILE is POSIX only",
)
def test_rich_file_logger_does_not_exhaust_fd(tmp_path: Path) -> None:
    """
    Regression-test for issue “Too many open files”.

    We artificially lower the soft RLIMIT_NOFILE to a small number, then start
    many threads that *simultaneously* write to the same `RichFileLogger`.
    With the **fixed** implementation only ONE file descriptor is ever open,
    so no thread raises `OSError: [Errno 24] Too many open files`.

    With the **old** implementation every call to `log()` did its own
    `open(..., "a")`, so the test reproducibly hit the limit and failed.
    """
    # Save & reduce the soft limit for the duration of the test
    soft_orig, hard_orig = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft_test = 32  # small number that we will overflow without the fix
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_test, hard_orig))

    try:
        log_file = str(tmp_path / "shared.log")
        threads: List[threading.Thread] = []
        errors: List[BaseException] = []
        starter = threading.Event()

        for _ in range(64):  # > soft_test to trigger exhaustion without fix
            t = threading.Thread(
                target=_stress_writer,
                args=(starter, errors, log_file),
                daemon=True,
            )
            threads.append(t)
            t.start()

        starter.set()  # let all threads proceed together

        for t in threads:
            t.join()

        if errors:
            # Show first error to aid debugging
            raise AssertionError(f"Errors occurred: {errors[0]!r}") from errors[0]

        # Additionally ensure the logger left no open fd when closed
        RichFileLogger(log_file, append=True, color=False).close()
        assert log_file not in RichFileLogger._instances  # type: ignore[attr-defined]
    finally:
        # Restore original limits to avoid side effects on other tests
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_orig, hard_orig))
