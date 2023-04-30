import logging


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
    import pkgutil
    import importlib

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        setup_logger(module.__name__, level)
