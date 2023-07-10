import sys
from typing import Any, Optional

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
