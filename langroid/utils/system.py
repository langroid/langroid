import difflib
import getpass
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import shutil
import socket
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DELETION_ALLOWED_PATHS = [
    ".qdrant",
    ".chroma",
    ".lancedb",
    ".weaviate",
]


def pydantic_major_version() -> int:
    try:
        pydantic_version = importlib.metadata.version("pydantic")
        major_version = int(pydantic_version.split(".")[0])
        return major_version
    except importlib.metadata.PackageNotFoundError:
        return -1


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


def hash(s: str) -> str:
    """
    Generate a SHA256 hash of a string.

    Args:
        s (str): The string to hash.

    Returns:
        str: The SHA256 hash of the string.
    """
    return update_hash(s=s)


def generate_unique_id() -> str:
    """Generate a unique ID using UUID4."""
    return str(uuid.uuid4())


def create_file(
    filepath: str | Path,
    content: str = "",
    if_exists: Literal["overwrite", "skip", "error", "append"] = "overwrite",
) -> None:
    """
    Create, overwrite or append to a file, with the given content
    at the specified filepath.
    If content is empty, it will simply touch to create an empty file.

    Args:
        filepath (str|Path): The relative path of the file to be created
        content (str): The content to be written to the file
        if_exists (Literal["overwrite", "skip", "error", "append"]):
            Action to take if file exists
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        if if_exists == "skip":
            logger.warning(f"File already exists, skipping: {filepath}")
            return
        elif if_exists == "error":
            raise FileExistsError(f"File already exists: {filepath}")
        elif if_exists == "append":
            mode = "a"
        else:  # overwrite
            mode = "w"
    else:
        mode = "w"

    if content == "" and mode in ["a", "w"]:
        filepath.touch()
        logger.warning(f"Empty file created: {filepath}")
    else:
        # the newline = '\n` argument is used to ensure that
        # newlines in the content are written as actual line breaks
        with open(filepath, mode, newline="\n") as f:
            f.write(content)
        action = "appended to" if mode == "a" else "created/updated in"
        logger.warning(f"Content {action}: {filepath}")


def read_file(path: str, line_numbers: bool = False) -> str:
    """
    Read the contents of a file.

    Args:
        path (str): The path to the file to be read.
        line_numbers (bool, optional): If True, prepend line numbers to each line.
            Defaults to False.

    Returns:
        str: The contents of the file, optionally with line numbers.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # raise an error if the file does not exist
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    file = Path(path).expanduser()
    content = file.read_text()
    if line_numbers:
        lines = content.splitlines()
        numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)
    return content


def diff_files(file1: str, file2: str) -> str:
    """
    Find the diffs between two files, in unified diff format.
    """
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    differ = difflib.unified_diff(lines1, lines2, fromfile=file1, tofile=file2)
    diff_result = "".join(differ)
    return diff_result


def list_dir(path: str | Path) -> list[str]:
    """
    List the contents of a directory.

    Args:
        path (str): The path to the directory.

    Returns:
        list[str]: A list of the files and directories in the specified directory.
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    return [str(p) for p in dir_path.iterdir()]
