import pytest
from llmagent.utils.configuration import Settings


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="show intermediate details, e.g. for debug mode",
    )
    parser.addoption("--nc", action="store_true", default=False, help="don't use cache")
    parser.addoption("--3", action="store_true", default=False, help="use GPT-3.5")


@pytest.fixture
def test_settings(request) -> Settings:
    return Settings(
        debug=request.config.getoption("--show"),
        cache=not request.config.getoption("--nc"),
        gpt3_5=request.config.getoption("--3"),
    )
