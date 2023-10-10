import pytest

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
        default="",
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


@pytest.fixture
def test_settings(request) -> Settings:
    return Settings(
        debug=request.config.getoption("--show"),
        cache=not request.config.getoption("--nc"),
        cache_type=request.config.getoption("--ct"),
        gpt3_5=request.config.getoption("--3"),
        stream=not request.config.getoption("--ns"),
        nofunc=request.config.getoption("--nof"),
        chat_model=request.config.getoption("--m"),
    )
