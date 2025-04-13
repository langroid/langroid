import random
import threading
import time

import pytest

from langroid.utils.configuration import (
    Settings,
    set_global,
    settings,
    temporary_settings,
    update_global_settings,
)


def test_update_global_settings():
    """
    Test that we can dynamically update the global settings object.
    """
    set_global(Settings(debug=True))
    assert settings.debug is True

    set_global(Settings(debug=False))
    assert settings.debug is False


# Shared list to collect exceptions
thread_exceptions = []


def safe_worker(target, *args, **kwargs):
    """Run a worker and capture any exception."""
    try:
        target(*args, **kwargs)
    except Exception as e:
        thread_exceptions.append(e)


def writer_worker(worker_id: int, iterations: int = 100):
    for i in range(iterations):
        new_debug = i % 2 == 0
        new_cfg = Settings(debug=new_debug)
        update_global_settings(new_cfg, keys=["debug"])
        time.sleep(random.uniform(0, 0.001))


def reader_worker(worker_id: int, read_list: list, iterations: int = 100):
    for _ in range(iterations):
        read_list.append(settings.debug)
        time.sleep(random.uniform(0, 0.001))


def context_worker(iterations: int = 50):
    for _ in range(iterations):
        orig_quiet = settings.quiet  # reads global value
        temp = Settings(quiet=True)
        with temporary_settings(temp):
            assert settings.quiet is True
            time.sleep(random.uniform(0, 0.001))
        # After the context, the calling thread’s settings revert to the global default.
        # Since temporary_settings is thread‑local,
        # concurrent updates do not affect this thread’s view.
        assert settings.quiet == orig_quiet


@pytest.mark.timeout(5)
def test_thread_safety():
    reader_results = []
    threads = []

    # Create threads and wrap targets with safe_worker
    for i in range(5):
        t = threading.Thread(target=safe_worker, args=(writer_worker, i))
        threads.append(t)

    for i in range(5):
        t = threading.Thread(
            target=safe_worker, args=(reader_worker, i, reader_results)
        )
        threads.append(t)

    for _ in range(2):
        t = threading.Thread(target=safe_worker, args=(context_worker,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Re-raise any exceptions captured
    if thread_exceptions:
        raise thread_exceptions[0]

    # Final consistency checks
    assert isinstance(settings.debug, bool)
    assert settings.quiet is False

    for val in reader_results:
        assert val in (True, False)


@pytest.mark.timeout(5)
def test_temporary_override_race():
    """
    This test forces two threads to use temporary_settings concurrently.
    Each thread:
      - Captures the original global value of settings.quiet.
      - Enters a temporary override (setting quiet=True).
      - Waits on a barrier until both threads are in the temporary context.
      - Exits the context and then records what settings.quiet evaluates to.

    In a proper thread‑safe implementation the final global value should still
    be the original (False), but in the old (non–thread‑safe) implementation a race
    condition between the two threads can result in one thread inadvertently leaving
    the global value set to True.
    """
    # Make sure global quiet is initially False.
    update_global_settings(Settings(quiet=False), keys=["quiet"])
    # Barrier for synchronizing two threads.
    barrier = threading.Barrier(2)
    # A place to record the final value of quiet after each thread exits its context.
    results = [None, None]

    def worker(index: int):
        # Define a temporary override that forces quiet=True.
        temp = Settings(quiet=True)
        with temporary_settings(temp):
            # While inside the context, the settings should be overridden.
            assert settings.quiet is True
            # Wait until both threads are here.
            barrier.wait()
            # Sleep briefly to let interleaving happen.
            time.sleep(0.01)
        # After the context, we expect the global setting to be restored.
        results[index] = settings.quiet
        # If a race occurred in the old implementation, the restored value may be wrong.

    threads = []
    for i in range(2):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Now, both threads should have seen the original value (False) restored.
    # In the broken implementation the race may cause one of these assertions to fail.
    assert (
        results[0] is False
    ), f"Thread 0 restored quiet={results[0]} instead of False."

    assert (
        results[1] is False
    ), f"Thread 1 restored quiet={results[1]} instead of False."
