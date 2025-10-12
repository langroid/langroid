import logging
import os
import threading

import pytest

from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig
from langroid.language_models import GeminiModel, OpenAIChatModel
from langroid.utils.configuration import Settings, set_global

logger = logging.getLogger(__name__)


def pytest_sessionfinish(session, exitstatus):
    """Hook to terminate pytest forcefully after displaying all test stats."""

    def terminate():
        if exitstatus == 0:
            print("All tests passed. Exiting cleanly.")
            os._exit(0)  # Exit code 0 for success
        else:
            print("Some tests failed. Exiting with error.")
            os._exit(1)  # Exit code 1 for error

    # Only set the timer if on GitHub Actions or another
    # CI environment where 'CI' is true
    if os.getenv("CI") == "true":
        threading.Timer(60, terminate).start()  # 60 seconds delay


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="show intermediate details, e.g. for debug mode",
    )
    parser.addoption("--nc", action="store_true", default=False, help="don't use cache")
    parser.addoption("--ns", action="store_true", default=False, help="no streaming")
    parser.addoption("--ct", default="redis", help="redis, fakeredis")
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
    parser.addoption(
        "--cross-encoder-device",
        action="store",
        default=None,
        help="Device for cross-encoder reranker (e.g. 'cpu', 'cuda', 'mps').",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fallback: mark test to use fallback models on failure"
    )


@pytest.fixture(scope="function")
def test_settings(request):
    base_settings = dict(
        debug=request.config.getoption("--show"),
        cache_type=request.config.getoption("--ct"),
        stream=not request.config.getoption("--ns"),
        max_turns=request.config.getoption("--turns"),
    )

    if request.node.get_closest_marker("fallback"):
        # we're in a test marked as requiring fallback,
        # so we re-run with a sequence of settings, mainly
        # on `chat_model` and `cache`.
        logger.warning("Running test with fallback settings")
        models = [request.config.getoption("--m")]
        if OpenAIChatModel.GPT4o not in models:
            # we may be using a weaker model, so add GPT4o as first fallback
            models.append(OpenAIChatModel.GPT4o)
        models.append(GeminiModel.GEMINI_2_FLASH)
        caches = [True] + [False] * (len(models) - 1)
        retry_count = getattr(request.node, "retry_count", 0)
        model = (
            models[retry_count]
            if retry_count < len(models)
            else request.config.getoption("--m")
        )
        cache = caches[retry_count] if retry_count < len(caches) else False
        logger.warning(f"Retry count: {retry_count}, model: {model}, cache: {cache}")
    else:
        model = request.config.getoption("--m")
        cache = not request.config.getoption("--nc")

    yield Settings(**base_settings, chat_model=model, cache=cache)


# Auto-inject this into every test, so we don't need to explicitly
# have `test_settings` as a parameter in every test function!
@pytest.fixture(autouse=True)
def auto_set_global_settings(test_settings):
    set_global(test_settings)
    yield


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
