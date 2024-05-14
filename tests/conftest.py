import os

import pytest

from langroid.language_models import OpenAIChatModel
from langroid.utils.configuration import Settings


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
    parser.addoption("--ct", default="redis", help="redis or momento")
    parser.addoption(
        "--m",
        default=OpenAIChatModel.GPT4o,
        help="""
        language model name, e.g. litellm/ollama/llama2, or 
        local or localhost:8000 or localhost:8000/v1
        """,
    )
    parser.addoption(
        "--nof",
        action="store_true",
        default=False,
        help="use model with no function_call",
    )


@pytest.fixture(scope="session")
def test_settings(request) -> Settings:
    chat_model = request.config.getoption("--m")
    if request.config.getoption("--3"):
        chat_model = OpenAIChatModel.GPT3_5_TURBO

    return Settings(
        debug=request.config.getoption("--show"),
        cache=not request.config.getoption("--nc"),
        cache_type=request.config.getoption("--ct"),
        gpt3_5=request.config.getoption("--3"),
        stream=not request.config.getoption("--ns"),
        chat_model=chat_model,
    )


@pytest.fixture(scope="session")
def redis_setup(redisdb):
    os.environ["REDIS_HOST"] = redisdb.connection_pool.connection_kwargs["host"]
    os.environ["REDIS_PORT"] = str(redisdb.connection_pool.connection_kwargs["port"])
    os.environ["REDIS_PASSWORD"] = ""  # Assuming no password for testing
    yield
    # Reset or clean up environment variables after tests
