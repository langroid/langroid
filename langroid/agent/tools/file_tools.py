from contextlib import chdir
from pathlib import Path
from textwrap import dedent
from typing import Callable, List, Tuple, Type

import git

from langroid.agent.tool_message import ToolMessage
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.pydantic_v1 import Field
from langroid.utils.git_utils import git_commit_file
from langroid.utils.system import create_file, list_dir, read_file


class ReadFileTool(ToolMessage):
    request: str = "read_file_tool"
    purpose: str = "Read the contents of a <file_path>"
    file_path: str

    _line_nums: bool = True  # whether to add line numbers to the content
    _curr_dir: Callable[[], str] | None = None

    @classmethod
    def create(
        cls,
        get_curr_dir: Callable[[], str] | None,
    ) -> Type["ReadFileTool"]:
        """
        Create a subclass of ReadFileTool for a specific directory

        Args:
            get_curr_dir (callable): A function that returns the current directory.

        Returns:
            Type[ReadFileTool]: A subclass of the ReadFileTool class, specifically
                for the current directory.
        """

        class CustomReadFileTool(cls):  # type: ignore
            _curr_dir: Callable[[], str] | None = (
                staticmethod(get_curr_dir) if get_curr_dir else None
            )

        return CustomReadFileTool

    @classmethod
    def examples(cls) -> List[ToolMessage | tuple[str, ToolMessage]]:
        return [
            cls(file_path="src/lib.rs"),
            (
                "I want to read the contents of src/main.rs",
                cls(file_path="src/main.rs"),
            ),
        ]

    def handle(self) -> str:
        # return contents as str for LLM to read
        # ASSUME: file_path should be relative to the curr_dir
        try:
            dir = (self._curr_dir and self._curr_dir()) or Path.cwd()
            with chdir(dir):
                # if file doesn't exist, return an error message
                content = read_file(self.file_path, self._line_nums)
            line_num_str = ""
            if self._line_nums:
                line_num_str = "(Line numbers added for reference only!)"
            return f""" 
    CONTENTS of {self.file_path}:
    {line_num_str}
    ---------------------------
    {content}
    """
        except FileNotFoundError:
            return f"File not found: {self.file_path}"


class WriteFileTool(XMLToolMessage):
    request: str = "write_file_tool"
    purpose: str = """
    Tool for writing <content> in a certain <language> to a <file_path>
    """

    file_path: str = Field(..., description="The path to the file to write the content")

    language: str = Field(
        default="",
        description="""
        The language of the content; could be human language or programming language
        """,
    )
    content: str = Field(
        ...,
        description="The content to write to the file",
        verbatim=True,  # preserve the content as is; uses CDATA section in XML
    )
    _curr_dir: Callable[[], str] | None = None
    _git_repo: Callable[[], git.Repo] | None = None
    _commit_message: str = "Agent write file tool"

    @classmethod
    def create(
        cls,
        get_curr_dir: Callable[[], str] | None,
        get_git_repo: Callable[[], str] | None,
    ) -> Type["WriteFileTool"]:
        """
        Create a subclass of WriteFileTool with the current directory and git repo.

        Args:
            get_curr_dir (callable): A function that returns the current directory.
            get_git_repo (callable): A function that returns the git repo.

        Returns:
            Type[WriteFileTool]: A subclass of the WriteFileTool class, specifically
                for the current directory and git repo.
        """

        class CustomWriteFileTool(cls):  # type: ignore
            _curr_dir: Callable[[], str] | None = (
                staticmethod(get_curr_dir) if get_curr_dir else None
            )
            _git_repo: Callable[[], str] | None = (
                staticmethod(get_git_repo) if get_git_repo else None
            )

        return CustomWriteFileTool

    @classmethod
    def examples(cls) -> List[ToolMessage | Tuple[str, ToolMessage]]:
        return [
            (
                """
                I want to define a simple hello world python function
                in a file "mycode/hello.py"
                """,
                cls(
                    file_path="mycode/hello.py",
                    language="python",
                    content="""
def hello():
    print("Hello, World!")
""",
                ),
            ),
            cls(
                file_path="src/lib.rs",
                language="rust",
                content="""
fn main() {
    println!("Hello, World!");
}                
""",
            ),
            cls(
                file_path="docs/intro.txt",
                content="""
# Introduction
This is the first sentence of the introduction.
                """,
            ),
        ]

    def handle(self) -> str:
        curr_dir = (self._curr_dir and self._curr_dir()) or Path.cwd()
        with chdir(curr_dir):
            create_file(self.file_path, self.content)
            msg = f"Content written to {self.file_path}"
            # possibly commit the file
            if self._git_repo:
                git_commit_file(
                    self._git_repo(),
                    self.file_path,
                    self._commit_message,
                )
                msg += " and committed"
        return msg


class ListDirTool(ToolMessage):
    request: str = "list_dir_tool"
    purpose: str = "List the contents of a <dir_path>"
    dir_path: str

    _curr_dir: Callable[[], str] | None = None

    @classmethod
    def create(
        cls,
        get_curr_dir: Callable[[], str] | None,
    ) -> Type["ReadFileTool"]:
        """
        Create a subclass of ListDirTool for a specific directory

        Args:
            get_curr_dir (callable): A function that returns the current directory.

        Returns:
            Type[ReadFileTool]: A subclass of the ReadFileTool class, specifically
                for the current directory.
        """

        class CustomListDirTool(cls):  # type: ignore
            _curr_dir: Callable[[], str] | None = (
                staticmethod(get_curr_dir) if get_curr_dir else None
            )

        return CustomListDirTool

    @classmethod
    def examples(cls) -> List[ToolMessage | tuple[str, ToolMessage]]:
        return [
            cls(dir_path="src"),
            (
                "I want to list the contents of src",
                cls(dir_path="src"),
            ),
        ]

    def handle(self) -> str:
        # ASSUME: dir_path should be relative to the curr_dir_path
        dir = (self._curr_dir and self._curr_dir()) or Path.cwd()
        with chdir(dir):
            contents = list_dir(self.dir_path)

        if not contents:
            return f"Directory not found or empty: {self.dir_path}"
        contents_str = "\n".join(contents)
        return dedent(
            f"""
            LISTING of directory {self.dir_path}:
            ---------------------------
            {contents_str}
            """.strip()
        )
