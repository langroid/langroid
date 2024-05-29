from typing import Any, Iterator

from _typeshed import Incomplete

from langroid.utils.configuration import settings as settings
from langroid.utils.constants import Colors as Colors

def shorten_text(text: str, chars: int = 40) -> str: ...
def print_long_text(
    color: str, style: str, preamble: str, text: str, chars: int | None = None
) -> None: ...
def show_if_debug(
    text: str,
    preamble: str,
    chars: int | None = None,
    color: str = "red",
    style: str = "italic",
) -> None: ...

class PrintColored:
    color: Incomplete
    def __init__(self, color: str) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

def silence_stdout() -> Iterator[None]: ...

class SuppressLoggerWarnings:
    logger: Incomplete
    original_level: Incomplete
    def __init__(self, logger: str | None = None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None: ...
