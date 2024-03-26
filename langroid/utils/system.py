import getpass
import hashlib
import importlib
import inspect
import logging
import shutil
import socket
import traceback
from typing import Any

logger = logging.getLogger(__name__)

DELETION_ALLOWED_PATHS = [
    ".qdrant",
    ".chroma",
    ".lancedb",
]


class LazyLoad:
    """Lazy loading of modules or classes."""

    def __init__(self, import_path: str) -> None:
        self.import_path = import_path
        self._target = None
        self._is_target_loaded = False

    def _load_target(self) -> None:
        if not self._is_target_loaded:
            try:
                # Attempt to import as a module
                self._target = importlib.import_module(self.import_path)  # type: ignore
            except ImportError:
                # If module import fails, attempt to import as a
                # class or function from a module
                module_path, attr_name = self.import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                self._target = getattr(module, attr_name)
            self._is_target_loaded = True

    def __getattr__(self, name: str) -> Any:
        self._load_target()
        return getattr(self._target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._load_target()
        if callable(self._target):
            return self._target(*args, **kwargs)
        else:
            raise TypeError(f"{self.import_path!r} object is not callable")


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


def friendly_error(e: Exception, msg: str = "An error occurred.") -> str:
    tb = traceback.format_exc()
    original_error_message: str = str(e)
    full_error_message: str = (
        f"{msg}\nOriginal error: {original_error_message}\nTraceback:\n{tb}"
    )
    return full_error_message


def generate_user_id(org: str = "") -> str:
    """
    Generate a unique user ID based on the username and machine name.
    Returns:
    """
    # Get the username
    username = getpass.getuser()

    # Get the machine's name
    machine_name = socket.gethostname()

    org_pfx = f"{org}_" if org else ""

    # Create a consistent unique ID based on the username and machine name
    unique_string = f"{org_pfx}{username}@{machine_name}"

    # Generate a SHA-256 hash of the unique string
    user_id = hashlib.sha256(unique_string.encode()).hexdigest()

    return user_id


def update_hash(hash: str | None = None, s: str = "") -> str:
    """
    Takes a SHA256 hash string and a new string, updates the hash with the new string,
    and returns the updated hash string.

    Args:
        hash (str): A SHA256 hash string.
        s (str): A new string to update the hash with.

    Returns:
        The updated hash in hexadecimal format.
    """
    # Create a new hash object if no hash is provided
    if hash is None:
        hash_obj = hashlib.sha256()
    else:
        # Convert the hexadecimal hash string to a byte object
        hash_bytes = bytes.fromhex(hash)
        hash_obj = hashlib.sha256(hash_bytes)

    # Update the hash with the new string
    hash_obj.update(s.encode("utf-8"))

    # Return the updated hash in hexadecimal format and the original string
    return hash_obj.hexdigest()
