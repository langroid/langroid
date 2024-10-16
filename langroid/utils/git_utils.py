import fnmatch
import logging
import textwrap
from pathlib import Path
from typing import List

import git
from github import Github, GithubException

from langroid.utils.system import create_file

logger = logging.getLogger(__name__)


def git_read_file(repo: str, filepath: str) -> str:
    """
    Read the contents of a file from a GitHub repository.

    Args:
        repo (str): The GitHub repository in the format "owner/repo"
        filepath (str): The file path relative to the repository root

    Returns:
        str: The contents of the file as a string
    """
    try:
        g = Github()
        github_repo = g.get_repo(repo)
        file_content = github_repo.get_contents(filepath)
        if isinstance(file_content, list) and len(file_content) > 0:
            return file_content[0].decoded_content.decode("utf-8")
        elif hasattr(file_content, "decoded_content"):
            return file_content.decoded_content.decode("utf-8")
        else:
            logger.error(f"Unexpected file_content type: {type(file_content)}")
            return ""
    except GithubException as e:
        logger.error(f"An error occurred while reading file {filepath}: {e}")
        return ""


def get_file_list(repo: str, dir: str, pat: str = "") -> List[str]:
    """
    Get a list of files in a specified directory of a GitHub repository.

    Args:
        repo (str): The GitHub repository in the format "owner/repo"
        dir (str): The directory path relative to the repository root
        pat (str): Optional wildcard pattern to filter file names (default: "")

    Returns:
        List[str]: A list of file paths in the specified directory
    """
    try:
        g = Github()
        github_repo = g.get_repo(repo)
        contents = github_repo.get_contents(dir)

        file_list = []
        if isinstance(contents, list):
            file_list = [content.path for content in contents if content.type == "file"]
        elif hasattr(contents, "path") and hasattr(contents, "type"):
            if contents.type == "file":
                file_list = [contents.path]

        if pat:
            file_list = [file for file in file_list if fnmatch.fnmatch(file, pat)]
        return sorted(file_list)

    except GithubException as e:
        logger.error(f"An error occurred while fetching file list: {e}")
        return []


def git_init_repo(dir: str) -> git.Repo | None:
    """
    Set up a Git repository in the specified directory.

    Args:
        dir (str): Path to the directory where the Git repository should be initialized

    Returns:
        git.Repo: The initialized Git repository object
    """
    repo_path = Path(dir).expanduser()
    try:
        repo = git.Repo.init(repo_path)
        logger.info(f"Git repository initialized in {repo_path}")

        gitignore_content = textwrap.dedent(
            """
        /target/
        **/*.rs.bk
        Cargo.lock
        """
        ).strip()

        gitignore_path = repo_path / ".gitignore"
        create_file(gitignore_path, gitignore_content)
        logger.info(f"Created .gitignore file in {repo_path}")

        # Ensure the default branch is 'main'
        # Check if we're on the master branch
        if repo.active_branch.name == "master":
            # Rename the branch
            repo.git.branch("-m", "master", "main")
            print("Branch renamed from 'master' to 'main'")
        else:
            print("Current branch is not 'master'. No changes made.")
        return repo
    except git.GitCommandError as e:
        logger.error(f"An error occurred while initializing the repository: {e}")
        return None


def git_commit_file(repo: git.Repo, filepath: str, msg: str) -> None:
    """
    Commit a file to a Git repository.

    Args:
        repo (git.Repo): The Git repository object
        filepath (str): Path to the file to be committed
        msg (str): The commit message

    Returns:
        None
    """
    try:
        repo.index.add([filepath])
        commit_msg = msg or f"Updated {filepath}"
        repo.index.commit(commit_msg)
        logger.info(f"Successfully committed {filepath}: {commit_msg}")
    except git.GitCommandError as e:
        logger.error(f"An error occurred while committing: {e}")


def git_commit_mods(repo: git.Repo, msg: str = "commit all changes") -> None:
    """
    Commit all modifications in the Git repository.
    Does not raise an error if there's nothing to commit.

    Args:
        repo (git.Repo): The Git repository object

    Returns:
        None
    """
    try:
        if repo.is_dirty():
            repo.git.add(update=True)
            repo.index.commit(msg)
            logger.info("Successfully committed all modifications")
        else:
            logger.info("No changes to commit")
    except git.GitCommandError as e:
        logger.error(f"An error occurred while committing modifications: {e}")


def git_restore_repo(repo: git.Repo) -> None:
    """
    Restore all unstaged, uncommitted changes in the Git repository.
    This function undoes any dirty files to the last commit.

    Args:
        repo (git.Repo): The Git repository object

    Returns:
        None
    """
    try:
        if repo.is_dirty():
            repo.git.restore(".")
            logger.info("Successfully restored all unstaged changes")
        else:
            logger.info("No unstaged changes to restore")
    except git.GitCommandError as e:
        logger.error(f"An error occurred while restoring changes: {e}")


def git_restore_file(repo: git.Repo, file_path: str) -> None:
    """
    Restore a specific file in the Git repository to its state in the last commit.
    This function undoes changes to the specified file.

    Args:
        repo (git.Repo): The Git repository object
        file_path (str): Path to the file to be restored

    Returns:
        None
    """
    try:
        repo.git.restore(file_path)
        logger.info(f"Successfully restored file: {file_path}")
    except git.GitCommandError as e:
        logger.error(f"An error occurred while restoring file {file_path}: {e}")


def git_create_checkout_branch(repo: git.Repo, branch: str) -> None:
    """
    Create and checkout a new branch in the given Git repository.
    If the branch already exists, it will be checked out.
    If we're already on the specified branch, no action is taken.

    Args:
        repo (git.Repo): The Git repository object
        branch (str): The name of the branch to create or checkout

    Returns:
        None
    """
    try:
        if repo.active_branch.name == branch:
            logger.info(f"Already on branch: {branch}")
            return

        if branch in repo.heads:
            repo.heads[branch].checkout()
            logger.info(f"Checked out existing branch: {branch}")
        else:
            new_branch = repo.create_head(branch)
            new_branch.checkout()
            logger.info(f"Created and checked out new branch: {branch}")
    except git.GitCommandError as e:
        logger.error(f"An error occurred while creating/checking out branch: {e}")


def git_diff_file(repo: git.Repo, filepath: str) -> str:
    """
    Show diffs of file between the latest commit and the previous one if any.

    Args:
        repo (git.Repo): The Git repository object
        filepath (str): Path to the file to be diffed

    Returns:
        str: The diff output as a string
    """
    try:
        # Get the two most recent commits
        commits = list(repo.iter_commits(paths=filepath, max_count=2))

        if len(commits) < 2:
            return "No previous commit found for comparison."

        # Get the diff between the two commits for the specific file
        diff = repo.git.diff(commits[1].hexsha, commits[0].hexsha, filepath)

        return str(diff)
    except git.GitCommandError as e:
        logger.error(f"An error occurred while getting diff: {e}")
        return f"Error: {str(e)}"
