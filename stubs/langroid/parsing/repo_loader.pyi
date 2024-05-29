from typing import Any

from _typeshed import Incomplete
from github.Label import Label
from pydantic import BaseModel, BaseSettings

from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document
from langroid.parsing.document_parser import (
    DocumentParser as DocumentParser,
)
from langroid.parsing.document_parser import (
    DocumentType as DocumentType,
)
from langroid.parsing.parser import Parser as Parser
from langroid.parsing.parser import ParsingConfig as ParsingConfig

logger: Incomplete

class IssueData(BaseModel):
    state: str
    year: int
    month: int
    day: int
    assignee: str | None
    size: str | None
    text: str

def get_issue_size(labels: list[Label]) -> str | None: ...

class RepoLoaderConfig(BaseSettings):
    non_code_types: list[str]
    file_types: list[str]
    exclude_dirs: list[str]

class RepoLoader:
    url: Incomplete
    config: Incomplete
    clone_path: Incomplete
    log_file: str
    repo: Incomplete
    def __init__(self, url: str, config: RepoLoaderConfig = ...) -> None: ...
    def get_issues(self, k: int | None = 100) -> list[IssueData]: ...
    def default_clone_path(self) -> str: ...
    def clone(self, path: str | None = None) -> str | None: ...
    def load_tree_from_github(
        self, depth: int, lines: int = 0
    ) -> dict[str, str | list[dict[str, Any]]]: ...
    def load(
        self, path: str | None = None, depth: int = 3, lines: int = 0
    ) -> tuple[dict[str, str | list[dict[str, Any]]], list[Document]]: ...
    @staticmethod
    def load_from_folder(
        path: str,
        depth: int = 3,
        lines: int = 0,
        file_types: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        url: str = "",
    ) -> tuple[dict[str, str | list[dict[str, Any]]], list[Document]]: ...
    @staticmethod
    def get_documents(
        path: str | bytes,
        parser: Parser = ...,
        file_types: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        depth: int = -1,
        lines: int | None = None,
        doc_type: str | DocumentType | None = None,
    ) -> list[Document]: ...
    def load_docs_from_github(
        self, k: int | None = None, depth: int | None = None, lines: int | None = None
    ) -> list[Document]: ...
    @staticmethod
    def select(
        structure: dict[str, str | list[dict[str, Any]]],
        includes: list[str],
        excludes: list[str] = [],
    ) -> dict[str, str | list[dict[str, Any]]]: ...
    @staticmethod
    def ls(structure: dict[str, str | list[dict]], depth: int = 0) -> list[str]: ...
    @staticmethod
    def list_files(
        dir: str,
        depth: int = 1,
        include_types: list[str] = [],
        exclude_types: list[str] = [],
    ) -> list[str]: ...
    @staticmethod
    def show_file_contents(tree: dict[str, str | list[dict[str, Any]]]) -> str: ...
