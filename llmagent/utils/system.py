import shutil
import logging

logger = logging.getLogger(__name__)

def rmdir(path) -> bool:
    """
    Remove a directory recursively.
    Args:
        path: path to directory to remove
    Returns:
        True if a dir was removed, false otherwise. Raises error if failed to remove.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        logger.warning(f"Directory '{path}' does not exist. No action taken.")
        return False
    except Exception as e:
        logger.error(f"Error while removing directory '{path}': {e}")
    return True

