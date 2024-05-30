import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Type

from rich import print as rprint
from rich.text import Text

from langroid.utils.configuration import settings
from langroid.utils.constants import Colors


def shorten_text(text: str, chars: int = 40) -> str:
    text = " ".join(text.split())
    return text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text


def print_long_text(
    color: str, style: str, preamble: str, text: str, chars: Optional[int] = None
) -> None:
    if chars is not None:
        text = " ".join(text.split())
        text = text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text
    styled_text = Text(text, style=style)
    rprint(f"[{color}]{preamble} {styled_text}")


def show_if_debug(
    text: str,
    preamble: str,
    chars: Optional[int] = None,
    color: str = "red",
    style: str = "italic",
) -> None:
    if settings.debug:
        print_long_text(color, style, preamble, text, chars)


class PrintColored:
    """Context to temporarily print in a desired color"""

    def __init__(self, color: str):
        self.color = color

    def __enter__(self) -> None:
        sys.stdout.write(self.color)
        sys.stdout.flush()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        print(Colors().RESET)


@contextmanager
def silence_stdout() -> Iterator[None]:
    """
    Temporarily silence all output to stdout and from rich.print.

    This context manager redirects all output written to stdout (which includes
    outputs from the built-in print function and rich.print) to /dev/null on
    UNIX-like systems or NUL on Windows. Once the context block exits, stdout is
    restored to its original state.

    Example:
        with silence_stdout_and_rich():
            print("This won't be printed")
            rich.print("This also won't be printed")

    Note:
        This suppresses both standard print functions and the rich library outputs.
    """
    platform_null = "/dev/null" if sys.platform != "win32" else "NUL"
    original_stdout = sys.stdout
    fnull = open(platform_null, "w")
    sys.stdout = fnull
    try:
        yield
    finally:
        sys.stdout = original_stdout
        fnull.close()


class SuppressLoggerWarnings:
    def __init__(self, logger: str | None = None):
        # If no logger name is given, get the root logger
        self.logger = logging.getLogger(logger)
        self.original_level = self.logger.getEffectiveLevel()

    def __enter__(self) -> None:
        # Set the logging level to 'ERROR' to suppress warnings
        self.logger.setLevel(logging.ERROR)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        # Reset the logging level to its original value
        self.logger.setLevel(self.original_level)
