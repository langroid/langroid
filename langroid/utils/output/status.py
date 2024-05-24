import logging
from contextlib import AbstractContextManager, ExitStack
from typing import Any

from rich.console import Console
from rich.errors import LiveError

from langroid.utils.configuration import quiet_mode, settings

console = Console()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def status(
    msg: str,
    log_if_quiet: bool = True,
) -> AbstractContextManager[Any]:
    """
    Displays a rich spinner if not in quiet mode, else optionally logs the message.
    """
    stack = ExitStack()
    logged = False
    if settings.quiet and log_if_quiet:
        logged = True
        logger.info(msg)

    if not settings.quiet:
        try:
            stack.enter_context(console.status(msg))
        except LiveError:
            if not logged:
                logger.info(msg)

    # When using rich spinner, we enforce quiet mode
    # (since output will be messy otherwise);
    # We make an exception to this when debug is enabled.
    stack.enter_context(quiet_mode(not settings.debug))

    return stack
