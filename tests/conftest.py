import pytest
from pydantic import BaseSettings

class TestOptions(BaseSettings):
    show: bool = False
    nocache: bool = False


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="show intermediate details, e.g. for debug mode"
    )
    parser.addoption(
        "--nc",
        action="store_true",
        default=False,
        help="don't use cache"
    )

@pytest.fixture
def options(request) -> TestOptions:
    return TestOptions(
        show=request.config.getoption("--show"),
        nocache=request.config.getoption("--nc"),
    )
