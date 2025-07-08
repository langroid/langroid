import logging
import os
import os.path
import sys
import threading
from typing import ClassVar, Dict, no_type_check

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
    handler = logging.FileHandler(filename, mode=file_mode, encoding="utf-8")
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
    """Singleton-per-path, ref-counted, thread-safe file logger.

    • Any number of calls to `RichFileLogger(path)` yield the same object.
    • A per-instance lock guarantees that the underlying file is opened only
      once, even when many threads construct the logger concurrently.
    • A reference counter tracks how many parts of the program are using the
      logger; the FD is closed only when the counter reaches zero.
    • All writes are serialised with a dedicated write-lock.
    """

    _instances: ClassVar[Dict[str, "RichFileLogger"]] = {}
    _ref_counts: ClassVar[Dict[str, int]] = {}
    # guards _instances & _ref_counts
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------ #
    # construction / destruction
    # ------------------------------------------------------------------ #
    def __new__(
        cls, log_file: str, append: bool = False, color: bool = True
    ) -> "RichFileLogger":
        with cls._class_lock:
            if log_file in cls._instances:
                cls._ref_counts[log_file] += 1
                return cls._instances[log_file]

            inst = super().__new__(cls)
            # create the per-instance init-lock *before* releasing class-lock
            inst._init_lock = threading.Lock()
            cls._instances[log_file] = inst
            cls._ref_counts[log_file] = 1
            return inst

    def __init__(self, log_file: str, append: bool = False, color: bool = True) -> None:
        # Double-checked locking: perform heavy init exactly once.
        if getattr(self, "_init_done", False):
            return

        if not hasattr(self, "_init_lock"):
            self._init_lock: threading.Lock = threading.Lock()

        with self._init_lock:
            if getattr(self, "_init_done", False):
                return

            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            mode = "a" if append else "w"
            self._owns_file: bool = True
            try:
                self.file = open(log_file, mode, buffering=1, encoding="utf-8")
            except OSError as exc:  # EMFILE: too many open files
                if exc.errno == 24:
                    # Fallback: reuse an already-open stream to avoid creating a new FD
                    self.file = sys.stderr
                    self._owns_file = False
                else:
                    raise
            self.log_file: str = log_file
            self.color: bool = color
            self.console: Console | None = (
                Console(file=self.file, force_terminal=True, width=200)
                if color
                else None
            )
            self._write_lock = threading.Lock()
            self._init_done = True  # set last

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @no_type_check
    def log(self, message: str) -> None:
        """Thread-safe write to the log file."""
        with self._write_lock:
            if self.color and self.console is not None:
                self.console.print(escape(message))
            else:
                print(message, file=self.file)
            self.file.flush()

    def close(self) -> None:
        """Decrease ref-count; close FD only when last user is done."""
        with self._class_lock:
            count = self._ref_counts.get(self.log_file, 0) - 1
            if count <= 0:
                self._ref_counts.pop(self.log_file, None)
                self._instances.pop(self.log_file, None)
                with self._write_lock:
                    if self._owns_file and not self.file.closed:
                        self.file.close()
            else:
                self._ref_counts[self.log_file] = count
