from llmagent.mytypes import Document
import logging
from github import Github, ContentFile
from dotenv import load_dotenv
import os
from typing import List, Union, Dict, Tuple
import itertools
from pydantic import BaseSettings
from collections import deque
import subprocess
import tempfile
import time
import json
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _get_decoded_content(content_file: ContentFile) -> str:
    if content_file.encoding == "base64":
        return content_file.decoded_content.decode("utf-8")
    elif content_file.encoding == "none":
        return content_file.content
    else:
        raise ValueError(f"Unsupported encoding: {content_file.encoding}")


class RepoLoaderConfig(BaseSettings):
    """
    Configuration for RepoLoader.
    """

    non_code_types: List[str] = [
        "md",
        "txt",
        "text",
    ]

    file_types: List[str] = [
        "py",
        "md",
        "yml",
        "yaml",
        "txt",
        "text",
        "sh",
        "ini",
        "toml",
        "cfg",
        "json",
        "rst",
        "Makefile",
        "Dockerfile",
    ]

    exclude_dirs: List[str] = [
        ".gitignore",
        ".gitmodules",
        ".gitattributes",
        ".git",
        ".idea",
        ".vscode",
        ".circleci",
    ]


class RepoLoader:
    """
    Class for recursively getting all file content in a repo.
    """

    def __init__(
        self,
        url: str,
        config: RepoLoaderConfig = RepoLoaderConfig(),
    ):
        """
        Args:
            url: full github url of repo, or just "owner/repo"
            file_types: list of file extensions to include
        """
        self.url = url
        self.config = config
        self.clone_path = None
        self.log_file = ".logs/repo_loader/download_log.json"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump({"junk": "ignore"}, f)
        with open(self.log_file, "r") as f:
            log = json.load(f)
        if self.url in log:
            logger.info(f"Repo Already downloaded in {log[self.url]}")
            self.clone_path = log[self.url]

        if "github.com" in self.url:
            repo_name = self.url.split("github.com/")[1]
        else:
            repo_name = self.url
        load_dotenv()
        # authenticated calls to github api have higher rate limit
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        g = Github(token)
        self.repo = self._get_repo_with_retry(g, repo_name)

    @staticmethod
    def _get_repo_with_retry(g: Github, repo_name: str, max_retries: int = 5):
        """
        Get a repo from the GitHub API, retrying if the request fails,
        with exponential backoff.

        Args:
            g:
            repo_name:
            max_retries:

        Returns:

        """
        base_delay = 2  # base delay in seconds
        max_delay = 60  # maximum delay in seconds

        for attempt in range(max_retries):
            try:
                return g.get_repo(repo_name)
            except Exception as e:
                delay = min(max_delay, base_delay * 2**attempt)
                logger.info(
                    f"Attempt {attempt+1} failed with error: {str(e)}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
        raise Exception(f"Failed to get repo {repo_name} after {max_retries} attempts.")

    def _get_dir_name(self) -> str:
        return urlparse(self.url).path.replace("/", "_")

    @staticmethod
    def _file_type(name: str) -> str:
        """
        Get the file type of a file name.
        Args:
            name: name of file, can be "a", "a.b", or ".b"
        Returns:
            str: file type; "a" => "a", "a.b" => "b", ".b" => "b"
            Examples:
                "Makefile" => "Makefile",
                "script.py" => "py",
                ".gitignore" => "gitignore"
        """
        # "a" -> ("a", ""), "a.b" -> ("a", ".b"), ".b" -> (".b", "")
        file_parts = os.path.splitext(name)
        if file_parts[1] == "":
            file_type = file_parts[0]  # ("a", "") => "a"
        else:
            file_type = file_parts[1][1:]  # (*,".b") => "b"
        return file_type

    def _is_code(self, file_type: str) -> bool:
        """
        Check if a file type is code.
        Args:
            file_type: file type, e.g. "py", "md", "txt"
        Returns:
            bool: whether file type is code
        """
        return file_type not in self.config.non_code_types

    def _is_allowed(self, content: ContentFile) -> bool:
        """
        Check if a file or directory content is allowed to be included.
        Args:
            content (ContentFile): The file or directory Content object.
        Returns:
            bool: Whether the file or directory is allowed to be included.
        """
        if content.type == "dir":
            return content.name not in self.config.exclude_dirs
        elif content.type == "file":
            return self._file_type(content.name) in self.config.file_types
        else:
            return False

    def default_clone_path(self):
        return tempfile.mkdtemp(suffix=self._get_dir_name())

    def clone(self, path: str = None) -> None:
        """
        Clone a GitHub repository to a local directory specified by `path`,
        if it has not already been cloned.

        Args:
            path (str): The local directory where the repository should be cloned.
                If not specified, a temporary directory will be created.
        """
        with open(self.log_file, "r") as f:
            log = json.load(f)

        if self.url in log and os.path.exists(log[self.url]):
            logger.warning(f"Repo Already downloaded in {log[self.url]}")
            self.clone_path = log[self.url]
            return self.clone_path

        self.clone_path = path
        if path is None:
            path = self.default_clone_path()
            self.clone_path = path

        try:
            subprocess.run(["git", "clone", self.url, path], check=True)
            log[self.url] = path
            with open(self.log_file, "w") as f:
                json.dump(log, f)
            return self.clone_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}")
        except Exception as e:
            logger.error(f"An error occurred while trying to clone the repository:{e}")

    def load_tree_from_github(
        self, depth: int, lines: int = 0
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
                Get a nested dictionary of GitHub repository file and directory names
                up to a certain depth, with file contents.

                Args:
                    depth (int): The depth level.
                    lines (int): The number of lines of file contents to include.

                Returns:
        Returns:
                    Dict[str, Union[str, List[Dict]]]:
                    A dictionary containing file and directory names, with file contents.
        """
        root_contents = self.repo.get_contents("")
        repo_structure = {
            "type": "dir",
            "name": "",
            "dirs": [],
            "files": [],
            "path": "",
        }

        # A queue of tuples (current_node, current_depth, parent_structure)
        queue = deque([(root_contents, 0, repo_structure)])

        while queue:
            current_node, current_depth, parent_structure = queue.popleft()

            for content in current_node:
                if not self._is_allowed(content):
                    continue
                if content.type == "dir" and current_depth < depth:
                    # Create a new sub-dictionary for this directory
                    new_dir = {
                        "type": "dir",
                        "name": content.name,
                        "dirs": [],
                        "files": [],
                        "path": content.path,
                    }
                    parent_structure["dirs"].append(new_dir)
                    queue.append(
                        (
                            self.repo.get_contents(content.path),
                            current_depth + 1,
                            new_dir,
                        )
                    )
                elif content.type == "file":
                    file_content = "\n".join(
                        _get_decoded_content(content).splitlines()[:lines]
                    )
                    file_dict = {
                        "type": "file",
                        "name": content.name,
                        "content": file_content,
                        "path": content.path,
                    }
                    parent_structure["files"].append(file_dict)

        return repo_structure

    def load(
        self,
        path: str = None,
        depth: int = 3,
        lines: int = 0,
    ) -> Tuple[Dict[str, Union[str, List[Dict]]], List[Document]]:
        """
        From a local folder `path` (if None, the repo clone path), get:
        - a nested dictionary (tree) of dicts, files and contents
        - a list of Document objects for each file.

        Args:
            path: The local folder path; if none, use self.clone_path()
            depth (int): The depth level.
            lines (int): The number of lines of file contents to include.

        Returns:
            Tuple of (dict, List_of_Documents):
            - A dictionary containing file and directory names, with file contents.
            - A list of Document objects for each file.
        """
        if path is None:
            if self.clone_path is None:
                self.clone()
            path = self.clone_path
        return self.load_from_folder(
            path=path,
            depth=depth,
            lines=lines,
            file_types=self.config.file_types,
            exclude_dirs=self.config.exclude_dirs,
            url=self.url,
        )

    @staticmethod
    def load_from_folder(
        path: str,
        depth: int = 3,
        lines: int = 0,
        file_types: List[str] = None,
        exclude_dirs: List[str] = None,
        url: str = "",
    ) -> Tuple[Dict[str, Union[str, List[Dict]]], List[Document]]:
        """
        From a local folder `path` (required), get:
        - a nested dictionary (tree) of dicts, files and contents, restricting to
            desired file_types and excluding undesired directories.
        - a list of Document objects for each file.

        Args:
            path (str): The local folder path, required.
            depth (int): The depth level. Optional, default 3.
            lines (int): The number of lines of file contents to include.
                    Optional, default 0 (no lines => empty string).
            file_types (List[str]): The file types to include.
                    Optional, default None (all).
            exclude_dirs (List[str]): The directories to exclude.
                    Optional, default None (no exclusions).
            url (str): Optional url, to be stored in docs as metadata. Default "".

        Returns:
            Tuple of (dict, List_of_Documents):
            - A dictionary containing file and directory names, with file contents.
            - A list of Document objects for each file.
        """

        folder_structure = {
            "type": "dir",
            "name": "",
            "dirs": [],
            "files": [],
            "path": "",
        }
        # A queue of tuples (current_path, current_depth, parent_structure)
        queue = deque([(path, 0, folder_structure)])
        docs = []
        exclude_dirs = exclude_dirs or []
        while queue:
            current_path, current_depth, parent_structure = queue.popleft()

            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                relative_path = os.path.relpath(item_path, path)
                if (os.path.isdir(item_path) and item in exclude_dirs) or (
                    os.path.isfile(item_path)
                    and file_types is not None
                    and RepoLoader._file_type(item) not in file_types
                ):
                    continue

                if os.path.isdir(item_path) and current_depth < depth:
                    # Create a new sub-dictionary for this directory
                    new_dir = {
                        "type": "dir",
                        "name": item,
                        "dirs": [],
                        "files": [],
                        "path": relative_path,
                    }
                    parent_structure["dirs"].append(new_dir)
                    queue.append((item_path, current_depth + 1, new_dir))
                elif os.path.isfile(item_path):
                    # Add the file to the current dictionary
                    with open(item_path, "r") as f:
                        file_lines = list(itertools.islice(f, lines))
                    file_content = "\n".join(line.strip() for line in file_lines)
                    if file_content == "":
                        continue

                    file_dict = {
                        "type": "file",
                        "name": item,
                        "content": file_content,
                        "path": relative_path,
                    }
                    parent_structure["files"].append(file_dict)
                    docs.append(
                        Document(
                            content=file_content,
                            metadata=dict(
                                repo=url,
                                source=relative_path,
                                url=url,
                                filename=item,
                                extension=RepoLoader._file_type(item),
                                language=RepoLoader._file_type(item),
                            ),
                        )
                    )
        return folder_structure, docs

    def load_docs_from_github(
        self, k: int = None, depth: int = None, lines: int = None
    ) -> List[Document]:
        """
        Directly from GitHub, recursively get all files in a repo that have one of the
        extensions, possibly up to a max number of files, max depth, and max number
        of lines per file (if any of these are specified).
        Args:
            k(int): max number of files to load, or None for all files
            depth(int): max depth to recurse, or None for infinite depth
            lines (int): max number of lines to get, from a file, or None for all lines
        Returns:
            list of Document objects, each has fields `content` and `metadata`,
            and `metadata` has fields `url`, `filename`, `extension`, `language`
        """
        contents = self.repo.get_contents("")
        stack = list(zip(contents, [0] * len(contents)))  # stack of (content, depth)
        # recursively get all files in repo that have one of the extensions
        docs = []
        i = 0

        while stack:
            if k is not None and i == k:
                break
            file_content, d = stack.pop()
            if not self._is_allowed(file_content):
                continue
            if file_content.type == "dir":
                if depth is None or d <= depth:
                    items = self.repo.get_contents(file_content.path)
                    stack.extend(list(zip(items, [d + 1] * len(items))))
            else:
                if depth is None or d <= depth:
                    # need to decode the file content, which is in bytes
                    text = _get_decoded_content(
                        self.repo.get_contents(file_content.path)
                    )
                    if lines is not None:
                        text = "\n".join(text.split("\n")[:lines])
                    i += 1

                    # Note `source` is important, it may be used to cite
                    # evidence for an answer.
                    # See  URLLoader
                    # TODO we should use Pydantic to enforce/standardize this

                    docs.append(
                        Document(
                            content=text,
                            metadata=dict(
                                repo=self.url,
                                source=file_content.html_url,
                                url=file_content.html_url,
                                filename=file_content.name,
                                extension=self._file_type(file_content.name),
                                language=self._file_type(file_content.name),
                            ),
                        )
                    )
        return docs

    @staticmethod
    def select(
        structure: Dict[str, Union[str, List[Dict]]],
        names: List[str],
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Filter a structure dictionary for certain directories and files.

        Args:
            structure (Dict[str, Union[str, List[Dict]]]): The structure dictionary.
            names (List[str]): A list of desired directory and file names.
            type (str): The type of the structure to filter for. If None, filter for
            both files and directories.
        Returns:
            Dict[str, Union[str, List[Dict]]]: The filtered structure dictionary.
        """
        filtered_structure = {
            "type": structure["type"],
            "name": structure["name"],
            "dirs": [],
            "files": [],
            "path": structure["path"],
        }

        for dir in structure["dirs"]:
            if dir["name"] in names:
                # If the directory is in the select list, include the whole subtree
                filtered_structure["dirs"].append(dir)
            else:
                # Otherwise, filter the directory's contents
                filtered_dir = RepoLoader.select(dir, names)
                if (
                    filtered_dir["dirs"] or filtered_dir["files"]
                ):  # only add if not empty
                    filtered_structure["dirs"].append(filtered_dir)

        for file in structure["files"]:
            if file["name"] in names:
                filtered_structure["files"].append(file)

        return filtered_structure

    @staticmethod
    def ls(structure: Dict[str, Union[str, List[Dict]]], depth: int = 0) -> List[str]:
        """
        Get a list of names of files or directories up to a certain depth from a
        structure dictionary.

        Args:
            structure (Dict[str, Union[str, List[Dict]]]): The structure dictionary.
            depth (int, optional): The depth level. Defaults to 0.

        Returns:
            List[str]: A list of names of files or directories.
        """
        names = []

        # A queue of tuples (current_structure, current_depth)
        queue = deque([(structure, 0)])

        while queue:
            current_structure, current_depth = queue.popleft()

            if current_depth <= depth:
                names.append(current_structure["name"])

                for dir in current_structure["dirs"]:
                    queue.append((dir, current_depth + 1))

                for file in current_structure["files"]:
                    # add file names only if depth is less than the limit
                    if current_depth < depth:
                        names.append(file["name"])
        names = [n for n in names if n not in ["", None]]
        return names
