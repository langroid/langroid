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


# Worker function to repeatedly update the global "debug" setting
def writer_worker(worker_id: int, iterations: int = 100):
    for i in range(iterations):
        new_debug = i % 2 == 0
        # Create a temporary Settings instance with the new debug value.
        new_cfg = Settings(debug=new_debug)
        # Update only the "debug" field
        update_global_settings(new_cfg, keys=["debug"])
        # Short sleep to simulate work and encourage thread interleaving.
        time.sleep(random.uniform(0, 0.001))


# Worker function to repeatedly read the global "debug" setting and record it
def reader_worker(worker_id: int, read_list: list, iterations: int = 100):
    for _ in range(iterations):
        # Read without locking because normal use should not change
        # in other parts of the library.
        read_list.append(settings.debug)
        time.sleep(random.uniform(0, 0.001))


# Worker function that uses the temporary_settings context manager to toggle "quiet"
def context_worker(iterations: int = 50):
    for _ in range(iterations):
        # Save the current state of "quiet"
        orig_quiet = settings.quiet
        # Create a temporary settings instance with quiet=True.
        temp = Settings(quiet=True)
        with temporary_settings(temp):
            # Inside context, "quiet" must be True.
            assert settings.quiet is True
            # Sleep a bit to simulate some processing.
            time.sleep(random.uniform(0, 0.001))
        # Once out of the context, "quiet" should be restored.
        assert settings.quiet == orig_quiet
        time.sleep(random.uniform(0, 0.001))


@pytest.mark.timeout(5)
def test_thread_safety():
    # Lists for reader threads to record observed values.
    reader_results = []

    # Create a number of threads for writers, readers, and temporary contexts.
    threads = []

    for i in range(5):
        t = threading.Thread(target=writer_worker, args=(i,))
        threads.append(t)

    for i in range(5):
        t = threading.Thread(target=reader_worker, args=(i, reader_results))
        threads.append(t)

    for _ in range(2):
        t = threading.Thread(target=context_worker)
        threads.append(t)

    # Start all threads.
    for t in threads:
        t.start()

    # Wait for all threads to finish.
    for t in threads:
        t.join()

    # At this point, no exceptions should have occurred.
    # We check that global settings remain in a consistent state.
    # For example, "debug" should be either True or False.
    assert isinstance(settings.debug, bool)
    # And by default "quiet" is False.
    assert settings.quiet is False

    # (Optionally) check that the reader threads captured only valid Boolean values.
    for val in reader_results:
        assert val in (True, False)
