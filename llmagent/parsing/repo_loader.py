from llmagent.mytypes import Document
import logging
from github import Github, ContentFile
from dotenv import load_dotenv
import os
from typing import List, Union, Dict
from pydantic import BaseSettings
from collections import deque
import subprocess

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

    file_types: List[str] = [
        "py", "md", "yml", "yaml", "txt", "text", "sh",
        "ini", "toml", "cfg", "json", "rst",
        "Makefile", "Dockerfile",
    ]

    dir_excludes: List[str] = [
        ".gitignore", ".gitmodules", ".gitattributes", ".git",
        ".idea", ".vscode", ".circleci"
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
        if "github.com" in self.url:
            repo_name = self.url.split("github.com/")[1]
        else:
            repo_name = self.url
        load_dotenv()
        # authenticated calls to github api have higher rate limit
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        g = Github(token)
        self.repo = g.get_repo(repo_name)


    def _file_type(self, name: str) -> str:
        """
        Get the file type of a file name.
        Args:
            name: name of file
        Returns:
            str: file type
        """
        # "a" -> ("a", ""), "a.b" -> ("a", ".b"), ".b" -> (".b", "")
        file_parts = os.path.splitext(name)
        if file_parts[1] == "":
            file_type = file_parts[0] # ("a", "") => "a"
        else:
            file_type = file_parts[1][1:] # (*,".b") => "b"
        return file_type

    def _is_allowed(self, content: ContentFile) -> bool:
        """
        Check if a file or directory content is allowed to be included.
        Args:
            content (ContentFile): The file or directory Content object.
        Returns:
            bool: Whether the file or directory is allowed to be included.
        """
        if content.type == "dir":
            return content.name not in self.config.dir_excludes
        elif content.type == "file":
            return self._file_type(content.name) in self.config.file_types
        else:
            return False

    def clone(self, path: str) -> None:
        """
        Clone a GitHub repository to a local directory specified by `path`.

        Args:
            path (str): The local directory where the repository should be cloned.
        """
        try:
            subprocess.run(['git', 'clone', self.url, path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e}")
        except Exception as e:
            print(f"An error occurred while trying to clone the repository: {e}")

    def get_repo_structure(
            self,
            depth: int,
            lines: int=0
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
        repo_structure = {"type": "dir", "name": "", "dirs": [], "files": []}

        # A queue of tuples (current_node, current_depth, parent_structure)
        queue = deque([(root_contents, 0, repo_structure)])

        while queue:
            current_node, current_depth, parent_structure = queue.popleft()

            for content in current_node:
                if not self._is_allowed(content):
                    continue
                if content.type == "dir" and current_depth < depth:
                    # Create a new sub-dictionary for this directory
                    new_dir = {"type": "dir",
                               "name": content.name,
                               "dirs": [],
                               "files": []}
                    parent_structure['dirs'].append(new_dir)
                    queue.append((self.repo.get_contents(content.path),
                                  current_depth + 1,
                                  new_dir))
                elif content.type == "file":
                    file_content = "\n".join(
                                 _get_decoded_content(content).
                                 splitlines()[:lines]
                             )
                    file_dict = {'type': 'file',
                                 'name': content.name,
                                 'content': file_content}
                    parent_structure['files'].append(file_dict)
                    #
                    # if lines > 0:
                    #     # Add the file to the current dictionary
                    #     file_content = "\n".join(
                    #         _get_decoded_content(content).
                    #         splitlines()[:lines]
                    #     )
                    #     file_dict = {'name': content.name, 'contents': file_content}
                    #     if 'files' in parent_structure:
                    #         parent_structure['files'].append(file_dict)
                    #     else:
                    #         parent_structure['files'] = [file_dict]
                    # else:
                    #     if 'files' in parent_structure:
                    #         parent_structure['files'].append(content.name)
                    #     else:
                    #         parent_structure['files'] = [content.name]

        return repo_structure

    def get_folder_structure(
            self,
            path:str,
            depth: int=3,
            lines: int=0
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Get a nested dictionary of local folder file and directory names
        up to a certain depth, with file contents.

        Args:
            depth (int): The depth level.
            lines (int): The number of lines of file contents to include.

        Returns:
            Dict[str, Union[str, List[Dict]]]:
            A dictionary containing file and directory names, with file contents.
        """
        folder_structure = {"type": "dir", "name": "", "dirs": [], "files": []}

        # A queue of tuples (current_path, current_depth, parent_structure)
        queue = deque([(path, 0, folder_structure)])

        while queue:
            current_path, current_depth, parent_structure = queue.popleft()

            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if ((os.path.isdir(item_path) and item in self.config.dir_excludes) or
                    (os.path.isfile(item_path) and
                     self._file_type(item) not in self.config.file_types)):
                    continue

                if os.path.isdir(item_path) and current_depth < depth:
                    # Create a new sub-dictionary for this directory
                    new_dir = {"type": "dir",
                               "name": item,
                               "dirs": [],
                               "files": []}
                    parent_structure['dirs'].append(new_dir)
                    queue.append((item_path, current_depth + 1, new_dir))
                elif os.path.isfile(item_path):
                    # Add the file to the current dictionary
                    try:
                        with open(item_path, 'r') as f:
                            file_content = "\n".join(
                                [next(f) for _ in range(lines)]
                            )
                    except StopIteration:
                        pass
                    file_dict = {'type': 'file', 'name': item, 'content': file_content}
                    parent_structure['files'].append(file_dict)

        return folder_structure


    def load(
            self, k: int = None, depth: int = None, lines: int = None
    ) -> List[Document]:
        """
        Recursively get all files in a repo that have one of the extensions,
        possibly up to a max number of files, max depth, and max number of lines per
        file (if any of these are specified).
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
                if (depth is None or d <= depth):
                    # need to decode the file content, which is in bytes
                    text = _get_decoded_content(self.repo.get_contents(
                        file_content.path))
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

    from typing import Dict, Union, List

    @staticmethod
    def select(
            structure: Dict[str, Union[str, List[Dict]]],
            names: List[str]
    ) -> Dict[str, Union[str, List[Dict]]]:
        """
        Filter a structure dictionary for certain directories and files.

        Args:
            structure (Dict[str, Union[str, List[Dict]]]): The structure dictionary.
            names (List[str]): A list of desired directory and file names.
        Returns:
            Dict[str, Union[str, List[Dict]]]: The filtered structure dictionary.
        """
        filtered_structure = {"type": structure["type"], "name": structure["name"],
                              "dirs": [], "files": []}

        for dir in structure["dirs"]:
            if dir["name"] in names:
                # If the directory is in the select list, include the whole subtree
                filtered_structure["dirs"].append(dir)
            else:
                # Otherwise, filter the directory's contents
                filtered_dir = RepoLoader.select(dir, names)
                if filtered_dir["dirs"] or filtered_dir["files"]:  # only add if not empty
                    filtered_structure["dirs"].append(filtered_dir)

        for file in structure["files"]:
            if file["name"] in names:
                filtered_structure["files"].append(file)


        return filtered_structure

