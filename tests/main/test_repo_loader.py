import ast
import json
import subprocess
from pathlib import Path

import pytest

from langroid.parsing.repo_loader import RepoLoader, RepoLoaderConfig


@pytest.mark.parametrize(
    "filename, content",
    [
        ("example.py", 'def greet():\n    return "hello"\n'),
        ("example.py", 'def greet():\n\treturn "hello"\n'),
        ("example.md", "First line  \nSecond line\n"),
        ("example.txt", "  indented text\n\nlast line  \n"),
    ],
)
def test_load_from_folder_preserves_line_whitespace(
    tmp_path: Path, filename: str, content: str
) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")

    tree, docs = RepoLoader.load_from_folder(str(tmp_path), lines=100)

    expected = content.rstrip("\n")
    assert tree["files"][0]["content"] == expected
    assert len(docs) == 1
    assert docs[0].content == expected
    assert docs[0].metadata.filename == filename
    if filename.endswith(".py"):
        ast.parse(docs[0].content)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_load_from_folder_preserves_whitespace_with_line_limit(
    tmp_path: Path, newline: str
) -> None:
    content = newline.join(["  first line  ", "\tsecond line", "third line", ""])
    (tmp_path / "example.txt").write_bytes(content.encode("utf-8"))

    tree, docs = RepoLoader.load_from_folder(str(tmp_path), lines=2)

    expected = "  first line  \n\tsecond line"
    assert tree["files"][0]["content"] == expected
    assert docs[0].content == expected


@pytest.mark.parametrize("content", ["", " \t\n", "plain text\n"])
def test_load_from_folder_retains_empty_file_filter(
    tmp_path: Path, content: str
) -> None:
    (tmp_path / "example.txt").write_text(content, encoding="utf-8")

    tree, docs = RepoLoader.load_from_folder(str(tmp_path), lines=100)

    if content.strip():
        assert tree["files"][0]["content"] == "plain text"
        assert docs[0].content == "plain text"
    else:
        assert tree["files"] == []
        assert docs == []


def test_repo_loader() -> None:
    """
    Test the RepoLoader class.
    """
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(url, config=RepoLoaderConfig())

    # directly create Document objects from github repo url
    # (uses many GitHub API calls, not recommended;
    #  use load() instead, which clones if needed, then loads from local folder)
    docs = repo_loader.load_docs_from_github(10, depth=0, lines=20)
    assert len(docs) > 0
    assert len(docs) <= 10
    for doc in docs:
        assert len(doc.content.split("\n")) <= 20

    # tree structure direct from github; again not recommended if easy to clone.
    tree = repo_loader.load_tree_from_github(depth=1, lines=3)
    assert len(tree) > 0

    # tree, docs from local clone (if exists, else clone first)
    tree, docs = repo_loader.load(depth=1, lines=5)
    assert len(tree) > 0
    assert len(docs) > 0, f"No docs loaded from repo {repo_loader.clone_path}"

    # test static fn that loads from a local folder;
    # this is a general fn that can be used to load from any folder,
    # not necessarily a git repo, or not necessarily even code, e.g.,
    # could be any folder of text files
    tree, docs = RepoLoader.load_from_folder(
        repo_loader.clone_path,
        depth=1,
        lines=5,
        file_types=["md", "txt", "toml"],
        exclude_dirs=[".git", "tests"],
    )
    assert len(tree) > 0
    assert len(docs) > 0

    # use a different fn to just load documents from folder
    docs = RepoLoader.get_documents(
        repo_loader.clone_path,
        depth=1,
        lines=5,
        file_types=["md", "txt", "toml"],
        exclude_dirs=[".git", "tests"],
    )
    assert len(docs) > 0

    # test making doc from single file path
    docs = RepoLoader.get_documents(
        Path(repo_loader.clone_path) / "pyproject.toml",
        depth=1,
        lines=5,
        file_types=["md", "txt", "toml"],
        exclude_dirs=[".git", "tests"],
    )
    assert len(docs) == 1

    # list all names to depth 2
    # Useful to provide LLM a listing of contents of a repo
    listing = repo_loader.ls(tree, depth=2)
    assert len(listing) > 0

    # dump to json
    s = json.dumps(tree, indent=2)
    assert len(s) > 0

    # select specific files
    desired = ["workflows", "Makefile", "pyproject.toml"]
    subtree = RepoLoader.select(tree, includes=desired)

    assert len(subtree["dirs"]) + len(subtree["files"]) <= 3

    # select non-existent files
    subtree = RepoLoader.select(tree, includes=["non-existent-file"])

    assert len(subtree["dirs"]) + len(subtree["files"]) == 0

    # list all names to depth 2
    listing = repo_loader.ls(tree, depth=2)
    assert len(listing) > 0


class _CloneFailed(subprocess.CalledProcessError):
    pass


def _loader(monkeypatch, tmp_path, url="https://github.com/nonexistent-org/nope"):
    """A RepoLoader that never touches GitHub: only the clone step is exercised."""
    from langroid.parsing import repo_loader as repo_loader_module
    from langroid.parsing.repo_loader import RepoLoader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(RepoLoader, "_get_repo_with_retry", lambda *a, **k: None)
    loader = RepoLoader(url)
    loader.clone_path = None
    return loader, repo_loader_module


def test_clone_returns_none_when_git_clone_fails(monkeypatch, tmp_path):
    """
    `clone` is typed `Optional[str]` and `load()` guards on `path is None`, so a
    failed clone must report None -- not the path it never cloned into.
    """
    loader, module = _loader(monkeypatch, tmp_path)

    def fail(*a, **k):
        raise _CloneFailed(1, ["git", "clone"])

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert loader.clone(str(tmp_path / "repo")) is None
    assert loader.clone_path is None


def test_load_reports_unclonable_repo(monkeypatch, tmp_path):
    """`load()` must raise its own 'Unable to clone repo' instead of loading nothing."""
    loader, module = _loader(monkeypatch, tmp_path)

    def fail(*a, **k):
        raise _CloneFailed(1, ["git", "clone"])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(ValueError, match="Unable to clone repo"):
        loader.load()


def test_clone_keeps_path_when_only_the_log_write_fails(monkeypatch, tmp_path):
    """A clone that succeeded must be reported even if bookkeeping fails.

    The download log is a cache for the next run. If writing it fails -- a
    read-only cache directory, say -- the broad `except Exception` used to
    swallow that too, so a complete repo sitting at `path` was reported as
    unclonable. The cost of a failed log write is re-cloning later, not
    losing the clone.
    """
    loader, module = _loader(monkeypatch, tmp_path)
    path = str(tmp_path / "repo")

    def ok(cmd, *a, **k):
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("hi\n")
        return subprocess.CompletedProcess(cmd, 0)

    real_open = module.open if hasattr(module, "open") else open

    def refuse_write(file, mode="r", *a, **k):
        if "w" in mode and str(file) == str(loader.log_file):
            raise PermissionError("read-only cache")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(module.subprocess, "run", ok)
    monkeypatch.setattr("builtins.open", refuse_write)

    assert loader.clone(path) == path
    assert loader.clone_path == path


def test_clone_returns_path_on_success(monkeypatch, tmp_path):
    """Control: a clone that succeeds still returns the path and records it."""
    loader, module = _loader(monkeypatch, tmp_path)

    def ok(cmd, *a, **k):
        # `git clone <url> <path>` -- create the destination so `_has_files` agrees
        dest = Path(cmd[-1])
        (dest / "sub").mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("hi\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", ok)
    path = str(tmp_path / "repo")
    assert loader.clone(path) == path
    assert loader.clone_path == path
