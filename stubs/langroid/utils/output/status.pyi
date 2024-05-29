from contextlib import AbstractContextManager
from typing import Any

from _typeshed import Incomplete

from langroid.utils.configuration import quiet_mode as quiet_mode
from langroid.utils.configuration import settings as settings

console: Incomplete
logger: Incomplete

def status(msg: str, log_if_quiet: bool = True) -> AbstractContextManager[Any]: ...
