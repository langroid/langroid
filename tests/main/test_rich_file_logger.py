from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List

import pytest

from langroid.utils.logging import RichFileLogger


def _make(path: str, n: int) -> List[RichFileLogger]:
    return [RichFileLogger(path, append=True, color=False) for _ in range(n)]


def _stress(start: threading.Event, errs: list[BaseException], path: str) -> None:
    start.wait()
    try:
        log = RichFileLogger(path, append=True, color=False)
        for _ in range(50):
            log.log("hi")
    except BaseException as exc:  # noqa: BLE001
        errs.append(exc)


@pytest.mark.parametrize("n", [1, 5, 50])
def test_singleton_and_fd(tmp_path: Path, n: int) -> None:
    file_path = tmp_path / "shared.log"
    loggers = _make(str(file_path), n)

    first = loggers[0]
    assert all(lg is first for lg in loggers)
    fd = first.file.fileno()
    assert all(lg.file.fileno() == fd for lg in loggers)

    first.log("one")

    # close once per acquisition ➜ final close closes fd
    for _ in range(n):
        first.close()
    assert first.file.closed


@pytest.mark.skipif(sys.platform.startswith("win"), reason="posix only")
def test_fd_limit(tmp_path: Path) -> None:
    import resource  # type: ignore

    soft0, hard0 = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, hard0))

    try:
        path = str(tmp_path / "stress.log")
        errs: list[BaseException] = []
        start = threading.Event()
        ths = [
            threading.Thread(target=_stress, args=(start, errs, path), daemon=True)
            for _ in range(64)
        ]
        for t in ths:
            t.start()
        start.set()
        for t in ths:
            t.join()
        if errs:
            raise AssertionError(f"Error: {errs[0]!r}") from errs[0]
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft0, hard0))



def test_write_after_peer_close(tmp_path: Path) -> None:
    """
    Scenario that used to raise `ValueError: I/O operation on closed file`.

    Two RichFileLogger handles are created for the same path. One is closed,
    the other continues to write. The test passes if no exception is raised.
    """
    log_path = tmp_path / "late_write.log"

    logger1 = RichFileLogger(str(log_path), append=True, color=False)
    logger2 = RichFileLogger(str(log_path), append=True, color=False)

    # Close the first handle
    logger1.close()

    # Second handle must still be functional
    logger2.log("log entry after peer close")

    # Clean up
    logger2.close()