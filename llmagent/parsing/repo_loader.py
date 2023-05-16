from llmagent.mytypes import Document
import logging
from github import Github, ContentFile
from dotenv import load_dotenv
import os
from typing import List


logger = logging.getLogger(__name__)


def _get_decoded_content(content_file: ContentFile) -> str:
    if content_file.encoding == "base64":
        return content_file.decoded_content.decode("utf-8")
    elif content_file.encoding == "none":
        return content_file.content
    else:
        raise ValueError(f"Unsupported encoding: {content_file.encoding}")


class RepoLoader:
    """
    Class for recursively getting all file content in a repo.
    """

    def __init__(
        self,
        url: str,
        extensions: List[str] = ["py", "md", "yml", "yaml", "txt", "text", "sh"],
    ):
        """
        Args:
            url: full github url of repo, or just "owner/repo"
            extensions:
        """
        self.url = url
        self.extensions = extensions

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
        if "github.com" in self.url:
            repo_name = self.url.split("github.com/")[1]
        else:
            repo_name = self.url
        load_dotenv()
        # authenticated calls to github api have higher rate limit
        token = os.getenv("GITHUB_ACCESS_TOKEN")
        g = Github(token)
        repo = g.get_repo(repo_name)
        contents = repo.get_contents("")
        stack = list(zip(contents, [0] * len(contents)))  # stack of (content, depth)
        # recursively get all files in repo that have one of the extensions
        docs = []
        i = 0

        while stack:
            if k is not None and i == k:
                break
            file_content, d = stack.pop()
            if file_content.type == "dir":
                if depth is None or d <= depth:
                    items = repo.get_contents(file_content.path)
                    stack.extend(list(zip(items, [d + 1] * len(items))))
            else:
                file_extension = os.path.splitext(file_content.name)[1][1:]
                if file_extension in self.extensions and (d <= depth or depth is None):
                    # need to decode the file content, which is in bytes
                    text = _get_decoded_content(repo.get_contents(file_content.path))
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
                                extension=file_extension,
                                language=file_extension,
                            ),
                        )
                    )
        return docs
