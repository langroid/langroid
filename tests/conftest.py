import os
import sys
import threading

import pytest

from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.language_models import OpenAIChatModel
from langroid.utils.configuration import Settings


def pytest_sessionfinish(session, exitstatus):
    """Hook to terminate pytest forcefully after displaying all test stats."""

    def force_exit(exit_code):
        # Attempt to terminate using sys.exit first
        sys.exit(exit_code)

        # If sys.exit fails, fall back to os._exit
        os._exit(exit_code)

    def terminate():
        # Log active threads for debugging
        print("Active threads at exit:")
        for thread in threading.enumerate():
            print(f"{thread.name} (daemon: {thread.isDaemon()})")

        # Exit based on test results
        if exitstatus == 0:
            print("All tests passed. Exiting cleanly.")
            force_exit(0)  # Exit code 0 for success
        else:
            print("Some tests failed. Exiting with error.")
            force_exit(1)  # Exit code 1 for error

    # Only set the timer if on GitHub Actions or another defined CI environment
    if "CI" in os.environ:
        timer = threading.Timer(60, terminate)  # 60 seconds delay
        timer.start()

        # Optionally, store the timer in the session for further management
        session._timeout_timer = timer


# Optional: Cleanup if needed
def pytest_unconfigure(config):
    if hasattr(config, "session") and hasattr(config.session, "_timeout_timer"):
        config.session._timeout_timer.cancel()


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="show intermediate details, e.g. for debug mode",
    )
    parser.addoption("--nc", action="store_true", default=False, help="don't use cache")
    parser.addoption("--3", action="store_true", default=False, help="use GPT-3.5")
    parser.addoption("--ns", action="store_true", default=False, help="no streaming")
    parser.addoption("--ct", default="redis", help="redis, fakeredis or momento")
    parser.addoption(
        "--m",
        default=OpenAIChatModel.GPT4o,
        help="""
        language model name, e.g. litellm/ollama/llama2, or 
        local or localhost:8000 or localhost:8000/v1
        """,
    )
    parser.addoption(
        "--turns",
        default=100,
        help="maximum number of turns in a task (to avoid inf loop)",
    )
    parser.addoption(
        "--nof",
        action="store_true",
        default=False,
        help="use model with no function_call",
    )
    # use multiple --first-test arguments to specify multiple tests to run first
    parser.addoption(
        "--first-test",
        action="append",
        default=[],
        help="Specify test FUNCTION(s) to run first.",
    )
    # use multiple --first-test-file arguments to specify multiple files to run first
    parser.addoption(
        "--first-test-file",
        action="append",
        default=[],
        help="Specify test FILE(s) to run first.",
    )


@pytest.fixture(scope="session")
def test_settings(request) -> Settings:
    chat_model = request.config.getoption("--m")
    max_turns = request.config.getoption("--turns")
    if request.config.getoption("--3"):
        chat_model = OpenAIChatModel.GPT3_5_TURBO

    return Settings(
        debug=request.config.getoption("--show"),
        cache=not request.config.getoption("--nc"),
        cache_type=request.config.getoption("--ct"),
        gpt3_5=request.config.getoption("--3"),
        stream=not request.config.getoption("--ns"),
        chat_model=chat_model,
        max_turns=max_turns,
    )


@pytest.fixture(scope="session")
def redis_setup(redisdb):
    os.environ["REDIS_HOST"] = redisdb.connection_pool.connection_kwargs["host"]
    os.environ["REDIS_PORT"] = str(redisdb.connection_pool.connection_kwargs["port"])
    os.environ["REDIS_PASSWORD"] = ""  # Assuming no password for testing
    yield
    # Reset or clean up environment variables after tests


def pytest_collection_modifyitems(config, items):
    # Get the lists of specified tests and files
    first_tests = config.getoption("--first-test")
    first_test_files = config.getoption("--first-test-file")

    priority_items = []
    other_items = list(items)  # Start with all items

    # Prioritize individual tests specified by --first-test
    for first_test in first_tests:
        current_priority_items = [
            item for item in other_items if first_test in item.nodeid
        ]
        priority_items.extend(current_priority_items)
        other_items = [
            item for item in other_items if item not in current_priority_items
        ]

    # Prioritize entire files specified by --first-test-file
    for first_test_file in first_test_files:
        current_priority_items = [
            item for item in other_items if first_test_file in str(item.fspath)
        ]
        priority_items.extend(current_priority_items)
        other_items = [
            item for item in other_items if item not in current_priority_items
        ]

    # Replace the items list with priority items first, followed by others
    items[:] = priority_items + other_items


@pytest.fixture(autouse=True)
def redis_close_connections():
    """Close all redis connections after each test fn, to avoid
    max connections exceeded error."""

    # Setup code here (if necessary)
    yield  # Yield to test execution
    # Cleanup code here
    redis = RedisCache(RedisCacheConfig(fake=False))
    try:
        redis.close_all_connections()
    except Exception:
        pass
