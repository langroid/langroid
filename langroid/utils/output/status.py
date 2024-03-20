import logging
from contextlib import ExitStack

from rich.console import Console

from langroid.utils.configuration import quiet_mode, settings

console = Console()
logger = logging.getLogger(__name__)


def status(
    msg: str,
    log_if_quiet: bool = True,
) -> ExitStack:
    """
    Displays a rich spinner if not in quiet mode, else optionally logs the message.
    """
    stack = ExitStack()

    if settings.quiet and log_if_quiet:
        logger.info(msg)
    else:
        stack.enter_context(console.status(msg))

    stack.enter_context(quiet_mode(not settings.debug))

    return stack
