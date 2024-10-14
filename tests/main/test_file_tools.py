import tempfile
from pathlib import Path
from typing import Callable

import pytest
from git import Repo

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.file_tools import ListDirTool, ReadFileTool, WriteFileTool
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.git_utils import git_init_repo


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def git_repo(temp_dir):
    repo = Repo.init(temp_dir)
    return repo


@pytest.fixture
def agent():
    cfg = ChatAgentConfig(
        name="test-write-file",
        vecdb=None,
        llm=OpenAIGPTConfig(),
        use_functions_api=False,
        use_tools=True,
        system_message=f"""
        When asked to read, write or do other operations on files,
        you MUST use one of the TOOLs:
        {ReadFileTool.default_value("request")},
        {WriteFileTool.default_value("request")},
        {ListDirTool.default_value("request")}
        Typically a file name or path will be provided, and you should
        NOT worry about what directory or path it is in. The TOOL will
        handle that for you.
        """,
    )
    return ChatAgent(cfg)


def test_write_file_tool(test_settings: Settings, temp_dir, git_repo, agent):
    set_global(test_settings)

    custom_write_file_tool = WriteFileTool.create(
        get_curr_dir=lambda: temp_dir, get_git_repo=lambda: git_repo
    )
    agent.enable_message(custom_write_file_tool)

    content = "print('Hello, World!')"
    file_path = "test_file.py"

    llm_msg = agent.llm_response_forget(
        f"Write a Python file named '{file_path}' with the content: {content}"
    )

    assert isinstance(agent.get_tool_messages(llm_msg)[0], custom_write_file_tool)

    agent_result = agent.handle_message(llm_msg).content
    assert f"Content written to {file_path}" in agent_result
    assert "and committed" in agent_result

    # Check if the file was created
    full_path = temp_dir / file_path
    assert full_path.exists()

    # Check if the content was written correctly
    with open(full_path, "r") as file:
        assert file.read().strip() == content

    # Check if the file was committed to the git repo
    assert not git_repo.is_dirty()
    assert file_path in git_repo.git.ls_files().split()


def test_write_file_tool_multiple_files(
    test_settings: Settings, temp_dir, git_repo, agent
):
    set_global(test_settings)

    custom_write_file_tool = WriteFileTool.create(
        get_curr_dir=lambda: temp_dir, get_git_repo=lambda: git_repo
    )
    agent.enable_message(custom_write_file_tool)

    files = [
        ("file1.txt", "This is file 1"),
        ("subdir/file2.py", "print('File 2')"),
        ("file3.md", "# File 3\nMarkdown content"),
    ]

    for file_path, content in files:
        full_path = temp_dir / file_path
        llm_msg = agent.llm_response_forget(
            f"Write a file named '{file_path}' with the content: {content}"
        )
        agent_result = agent.handle_message(llm_msg).content

        assert f"Content written to {file_path}" in agent_result
        assert "and committed" in agent_result

        assert full_path.exists()

        with open(full_path, "r") as file:
            assert file.read().strip() == content

        assert not git_repo.is_dirty()
        assert file_path in git_repo.git.ls_files().split()

    # Check if all files are in the repo
    assert len(git_repo.git.ls_files().split()) == len(files)


def test_write_file_tool_overwrite(test_settings: Settings, temp_dir, git_repo, agent):
    set_global(test_settings)

    custom_write_file_tool = WriteFileTool.create(
        get_curr_dir=lambda: temp_dir, get_git_repo=lambda: git_repo
    )
    agent.enable_message(custom_write_file_tool)

    file_path = "overwrite_test.txt"
    original_content = "Original content"
    new_content = "New content"

    # Write the original content
    llm_msg = agent.llm_response_forget(
        f"Write a file named '{file_path}' with the content: {original_content}"
    )
    agent.handle_message(llm_msg)

    # Overwrite with new content
    llm_msg = agent.llm_response_forget(
        f"Write a file named '{file_path}' with the content: {new_content}"
    )
    agent_result = agent.handle_message(llm_msg).content

    assert f"Content written to {file_path}" in agent_result
    assert "and committed" in agent_result

    full_path = temp_dir / file_path
    with open(full_path, "r") as file:
        assert file.read().strip() == new_content

    # Check git history
    commits = list(git_repo.iter_commits())
    assert len(commits) == 2


def test_read_file_tool(test_settings: Settings, temp_dir, agent):
    set_global(test_settings)

    custom_read_file_tool = ReadFileTool.create(get_curr_dir=lambda: temp_dir)
    agent.enable_message(custom_read_file_tool)

    # Create a test file
    file_path = "test_read.txt"
    content = "This is a test file content."
    with open(temp_dir / file_path, "w") as f:
        f.write(content)

    llm_msg = agent.llm_response_forget(f"Read the contents of the file '{file_path}'")

    assert isinstance(agent.get_tool_messages(llm_msg)[0], custom_read_file_tool)

    agent_result = agent.handle_message(llm_msg).content
    # there is just one line so no worries about line number
    assert content in agent_result


def test_read_file_tool_not_exist(test_settings: Settings, temp_dir, agent):
    set_global(test_settings)

    custom_read_file_tool = ReadFileTool.create(get_curr_dir=lambda: temp_dir)
    agent.enable_message(custom_read_file_tool)
    task = lr.Task(agent, interactive=False, done_if_response=[lr.Entity.AGENT])
    nonexistent_file = "nonexistent.txt"
    agent_result = task.run(f"Read the contents of the file '{nonexistent_file}'")
    assert "File not found" in agent_result.content


def test_read_file_tool_multiple_files(test_settings: Settings, temp_dir, agent):
    set_global(test_settings)

    custom_read_file_tool = ReadFileTool.create(get_curr_dir=lambda: temp_dir)
    agent.enable_message(custom_read_file_tool)

    files = [
        ("file1.txt", "Content of file 1"),
        ("subdir/file2.py", "print('Content of file 2')"),
        ("file3.md", "# File 3\nMarkdown content"),
    ]

    for file_path, content in files:
        full_path = temp_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

        llm_msg = agent.llm_response_forget(
            f"Read the contents of the file '{file_path}'"
        )
        agent_result = agent.handle_message(llm_msg).content

        # Check if each line of content is in the result, ignoring line numbers
        for line in content.split("\n"):
            assert any(line in result_line for result_line in agent_result.split("\n"))


def test_list_dir_tool(test_settings: Settings, temp_dir, agent):
    set_global(test_settings)

    custom_list_dir_tool = ListDirTool.create(get_curr_dir=lambda: temp_dir)
    agent.enable_message(custom_list_dir_tool)

    # Create some test files and directories
    (temp_dir / "file1.txt").touch()
    (temp_dir / "file2.py").touch()
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.md").touch()
    (temp_dir / "subdir" / "main.rs").touch()
    (temp_dir / "nulldir").mkdir()

    llm_msg = agent.llm_response_forget(
        "List the contents of the current directory '.' "
    )

    assert isinstance(agent.get_tool_messages(llm_msg)[0], custom_list_dir_tool)

    agent_result = agent.handle_message(llm_msg).content

    assert "file1.txt" in agent_result
    assert "file2.py" in agent_result
    assert "subdir" in agent_result

    llm_msg = agent.llm_response_forget(
        "List the contents of the current directory 'subdir' "
    )

    assert isinstance(agent.get_tool_messages(llm_msg)[0], custom_list_dir_tool)

    agent_result = agent.handle_message(llm_msg).content

    assert "file3.md" in agent_result
    assert "main.rs" in agent_result

    llm_msg = agent.llm_response_forget(
        "List the contents of the current directory 'nulldir' "
    )
    assert isinstance(agent.get_tool_messages(llm_msg)[0], custom_list_dir_tool)

    agent_result = agent.handle_message(llm_msg).content

    assert "empty" in agent_result


@pytest.fixture
def my_write_file_tool(temp_dir):
    git_repo = git_init_repo(temp_dir)

    def temp_dir_fn():
        return temp_dir

    def git_repo_fn():
        return git_repo

    class MyWriteFileTool(WriteFileTool):
        _curr_dir: Callable[[], str] = staticmethod(temp_dir_fn)
        _git_repo: Callable[[], Repo] = staticmethod(git_repo_fn)

    return MyWriteFileTool


def test_my_write_file_tool(
    test_settings: Settings, temp_dir, my_write_file_tool, agent
):
    set_global(test_settings)

    git_repo = git_init_repo(temp_dir)
    agent.enable_message(my_write_file_tool)

    content = "print('Hello from MyWriteFileTool')"
    file_path = "test_my_file.py"

    llm_msg = agent.llm_response_forget(
        f"Write a Python file named '{file_path}' with the content: {content}"
    )

    assert isinstance(agent.get_tool_messages(llm_msg)[0], my_write_file_tool)

    agent_result = agent.handle_message(llm_msg).content
    assert f"Content written to {file_path}" in agent_result
    assert "and committed" in agent_result

    full_path = temp_dir / file_path
    assert full_path.exists()

    with open(full_path, "r") as file:
        assert file.read().strip() == content

    assert not git_repo.is_dirty()
    assert file_path in git_repo.git.ls_files().split()
