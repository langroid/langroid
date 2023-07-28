import logging
import os.path
from typing import no_type_check

import colorlog
from rich.console import Console


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


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
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
    if not logger.hasHandlers():
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
    if not append:
        if os.path.exists(filename):
            os.remove(filename)

    logger = setup_logger(name)
    handler = logging.FileHandler(filename)
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
    def __init__(self, log_file: str, append: bool = False, color: bool = True):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        if not append:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
        self.file = None
        self.console = None
        self.append = append
        self.color = color

    @no_type_check
    def log(self, message: str) -> None:
        with open(self.log_file, "a") as f:
            if self.color:
                console = Console(file=f, force_terminal=True, width=200)
                console.print(message)
            else:
                print(message, file=f)
