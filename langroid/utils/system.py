import inspect
import logging
import shutil

logger = logging.getLogger(__name__)

DELETION_ALLOWED_PATHS = [
    ".qdrant",
    ".chroma",
]


def rmdir(path: str) -> bool:
    """
    Remove a directory recursively.
    Args:
        path (str): path to directory to remove
    Returns:
        True if a dir was removed, false otherwise. Raises error if failed to remove.
    """
    if not any([path.startswith(p) for p in DELETION_ALLOWED_PATHS]):
        raise ValueError(
            f"""
        Removing Dir '{path}' not allowed. 
        Must start with one of {DELETION_ALLOWED_PATHS}
        This is a safety measure to prevent accidental deletion of files.
        If you are sure you want to delete this directory, please add it 
        to the `DELETION_ALLOWED_PATHS` list in langroid/utils/system.py and 
        re-run the command.
        """
        )

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        logger.warning(f"Directory '{path}' does not exist. No action taken.")
        return False
    except Exception as e:
        logger.error(f"Error while removing directory '{path}': {e}")
    return True


def caller_name() -> str:
    """
    Who called the function?
    """
    frame = inspect.currentframe()
    if frame is None:
        return ""

    caller_frame = frame.f_back

    # If there's no caller frame, the function was called from the global scope
    if caller_frame is None:
        return ""

    return caller_frame.f_code.co_name
