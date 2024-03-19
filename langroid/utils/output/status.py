import logging
from contextlib import nullcontext

from rich.console import Console
from rich.status import Status

from langroid.utils.configuration import settings

console = Console()
logger = logging.getLogger(__name__)


def status(
    msg: str,
    log_if_quiet: bool = True,
) -> Status | nullcontext[None]:
    if settings.quiet:
        if log_if_quiet:
            logger.info(msg)

        return nullcontext()
    else:
        return console.status(msg)
