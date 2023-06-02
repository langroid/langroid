import logging

import colorlog


# Define a function to set up the colored logger
def setup_colored_logging():
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


def setup_logger(name: str, level: int):
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
