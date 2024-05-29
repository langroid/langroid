from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import printing
    from .printing import (
        shorten_text,
        print_long_text,
        show_if_debug,
        SuppressLoggerWarnings,
        PrintColored,
    )
    from .status import status
else:
    printing = LazyLoad("langroid.utils.output.printing")
    shorten_text = LazyLoad("langroid.utils.output.printing.shorten_text")
    print_long_text = LazyLoad("langroid.utils.output.printing.print_long_text")
    show_if_debug = LazyLoad("langroid.utils.output.printing.show_if_debug")
    SuppressLoggerWarnings = LazyLoad(
        "langroid.utils.output.printing.SuppressLoggerWarnings"
    )
    PrintColored = LazyLoad("langroid.utils.output.printing.PrintColored")
    status = LazyLoad("langroid.utils.output.status")


__all__ = [
    "printing",
    "shorten_text",
    "print_long_text",
    "show_if_debug",
    "SuppressLoggerWarnings",
    "PrintColored",
    "status",
]
