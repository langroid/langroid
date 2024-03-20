import logging
from contextlib import AbstractContextManager, ExitStack
from typing import Any

from rich.console import Console

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

    if settings.quiet:
        if log_if_quiet:
            logger.info(msg)
    if settings.quiet and log_if_quiet:
        logger.info(msg)
    else:
        stack.enter_context(console.status(msg))

    stack.enter_context(quiet_mode(not settings.debug))

    return stack
