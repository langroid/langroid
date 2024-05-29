import logging

from _typeshed import Incomplete

def setup_colored_logging() -> None: ...
def setup_logger(
    name: str, level: int = ..., terminal: bool = False
) -> logging.Logger: ...
def setup_console_logger(name: str) -> logging.Logger: ...
def setup_file_logger(
    name: str,
    filename: str,
    append: bool = False,
    log_format: bool = False,
    propagate: bool = False,
) -> logging.Logger: ...
def setup_loggers_for_package(package_name: str, level: int) -> None: ...

class RichFileLogger:
    log_file: Incomplete
    file: Incomplete
    console: Incomplete
    append: Incomplete
    color: Incomplete
    def __init__(
        self, log_file: str, append: bool = False, color: bool = True
    ) -> None: ...
    def log(self, message) -> None: ...
