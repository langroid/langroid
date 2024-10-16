import os
from unittest.mock import patch

import pytest
from github import GithubException

from langroid.utils.git_utils import (
    get_file_list,
    git_commit_file,
    git_commit_mods,
    git_create_checkout_branch,
    git_diff_file,
    git_init_repo,
    git_read_file,
    git_restore_file,
    git_restore_repo,
)


@pytest.fixture
def mock_github():
    with patch("langroid.utils.git_utils.Github") as mock:
        yield mock


@pytest.fixture
def temp_git_repo(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo = git_init_repo(str(repo_path))
    return repo


def test_git_read_file(mock_github):
    mock_content = (
        mock_github.return_value.get_repo.return_value.get_contents.return_value
    )
    mock_content.decoded_content = b"test content"

    content = git_read_file("owner/repo", "test.txt")
    assert content == "test content"


def test_git_read_file_exception(mock_github):
    mock_github.return_value.get_repo.side_effect = GithubException(
        404, "Not Found", {}
    )

    content = git_read_file("owner/repo", "test.txt")
    assert content == ""


def test_get_file_list(mock_github):
    mock_content = [
        type("obj", (), {"path": "file1.txt", "type": "file"})(),
        type("obj", (), {"path": "file2.md", "type": "file"})(),
    ]
    mock_github.return_value.get_repo.return_value.get_contents.return_value = (
        mock_content
    )

    files = get_file_list("owner/repo", "dir", "*.txt")
    assert files == ["file1.txt"]


def test_git_init_repo(temp_git_repo):
    assert temp_git_repo is not None
    assert os.path.exists(os.path.join(temp_git_repo.working_dir, ".gitignore"))


def test_git_commit_file(temp_git_repo):
    test_file = os.path.join(temp_git_repo.working_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Test content")

    git_commit_file(temp_git_repo, "test.txt", "Test commit")

    assert "test.txt" in temp_git_repo.git.ls_files().split()


def test_git_commit_mods(temp_git_repo):
    test_file = os.path.join(temp_git_repo.working_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Test content")

    # Add the file to the git index
    temp_git_repo.index.add([test_file])

    git_commit_mods(temp_git_repo)

    assert "test.txt" in temp_git_repo.git.ls_files().split()

    # Check if the file was actually committed
    assert len(temp_git_repo.head.commit.tree.blobs) > 0
    assert any(blob.name == "test.txt" for blob in temp_git_repo.head.commit.tree.blobs)


def test_git_restore_repo(temp_git_repo):
    test_file = os.path.join(temp_git_repo.working_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Test content")

    git_restore_repo(temp_git_repo)

    assert "test.txt" not in temp_git_repo.git.ls_files().split()


def test_git_restore_file(temp_git_repo):
    test_file = os.path.join(temp_git_repo.working_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Initial content")
    git_commit_file(temp_git_repo, "test.txt", "Initial commit")

    with open(test_file, "w") as f:
        f.write("Modified content")

    git_restore_file(temp_git_repo, "test.txt")

    with open(test_file, "r") as f:
        content = f.read()
    assert content == "Initial content"


def test_git_create_checkout_branch(temp_git_repo):
    # Make an initial commit to create the HEAD reference
    test_file = os.path.join(temp_git_repo.working_dir, "initial.txt")
    with open(test_file, "w") as f:
        f.write("Initial content")
    temp_git_repo.index.add([test_file])
    temp_git_repo.index.commit("Initial commit")

    git_create_checkout_branch(temp_git_repo, "new-branch")
    assert temp_git_repo.active_branch.name == "new-branch"


def test_git_diff_file(temp_git_repo):
    test_file = os.path.join(temp_git_repo.working_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Initial content")
    git_commit_file(temp_git_repo, "test.txt", "Initial commit")

    with open(test_file, "w") as f:
        f.write("Modified content")
    git_commit_file(temp_git_repo, "test.txt", "Modified commit")

    diff = git_diff_file(temp_git_repo, "test.txt")
    assert "Initial content" in diff
    assert "Modified content" in diff
