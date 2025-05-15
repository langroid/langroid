import logging
import os
import os.path
import threading
from typing import Dict, no_type_check

import colorlog
from rich.console import Console
from rich.markup import escape


# Define a function to set up the colored logger
def setup_colored_logging() -> None:
    # Define the log format with color codes
    log_format = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s"
    # Create a color formatter
    color_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    # Configure the root logger to use the color formatter
    handler = logging.StreamHandler()
    handler.setFormatter(color_formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    # logger.setLevel(logging.DEBUG)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    terminal: bool = False,
) -> logging.Logger:
    """
    Set up a logger of module `name` at a desired level.
    Args:
        name: module name
        level: desired logging level
    Returns:
        logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers() and terminal:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_console_logger(name: str) -> logging.Logger:
    logger = setup_logger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_file_logger(
    name: str,
    filename: str,
    append: bool = False,
    log_format: bool = False,
    propagate: bool = False,
) -> logging.Logger:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_mode = "a" if append else "w"
    logger = setup_logger(name, terminal=False)
    handler = logging.FileHandler(filename, mode=file_mode)
    handler.setLevel(logging.INFO)
    if log_format:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger


def setup_loggers_for_package(package_name: str, level: int) -> None:
    """
    Set up loggers for all modules in a package.
    This ensures that log-levels of modules outside the package are not affected.
    Args:
        package_name: main package name
        level: desired logging level
    Returns:
    """
    import importlib
    import pkgutil

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        setup_logger(module.__name__, level)


class RichFileLogger:
    """Thread-safe, singleton-per-file logger.

    • Only ONE RichFileLogger instance – and therefore one open FD – exists for
      any given log-file path across all threads/tasks.
    • Instance creation and first-time initialisation are protected against
      races, so the file is opened exactly once.
    """

    _instances: Dict[str, "RichFileLogger"] = {}
    _class_lock = threading.Lock()  # guards _instances map

    # --------------------------------------------------------------------- #
    # construction / initialisation
    # --------------------------------------------------------------------- #
    def __new__(
        cls,
        log_file: str,
        append: bool = False,
        color: bool = True,
    ) -> "RichFileLogger":
        with cls._class_lock:
            if log_file in cls._instances:
                return cls._instances[log_file]
            inst = super().__new__(cls)
            cls._instances[log_file] = inst
            return inst

    def __init__(self, log_file: str, append: bool = False, color: bool = True) -> None:
        # Double-checked locking: do expensive work exactly once.
        if getattr(self, "_init_done", False):
            return
        # Each instance has its own init-lock so competing threads that obtained
        # the same (not-yet-initialised) object serialise inside __init__.
        self._init_lock = getattr(self, "_init_lock", threading.Lock())
        with self._init_lock:
            if getattr(self, "_init_done", False):
                return  # another thread finished initialising while we waited

            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            mode = "a" if append else "w"
            self.file = open(log_file, mode, buffering=1, encoding="utf-8")
            self.log_file: str = log_file
            self.color: bool = color
            self.console: Console | None = (
                Console(file=self.file, force_terminal=True, width=200)
                if color
                else None
            )
            self._write_lock = threading.Lock()  # guards writes
            self._init_done = True

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    @no_type_check
    def log(self, message: str) -> None:
        """Write `message` to the log file in a thread-safe manner."""
        with self._write_lock:
            if self.color and self.console is not None:
                self.console.print(escape(message))
            else:
                print(message, file=self.file)
            self.file.flush()

    def close(self) -> None:
        """Close the FD and forget the singleton instance for this path."""
        with self._write_lock:
            if not self.file.closed:
                self.file.close()
        with self._class_lock:
            self._instances.pop(self.log_file, None)
