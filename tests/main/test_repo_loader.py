import json
from pathlib import Path

from langroid.parsing.repo_loader import RepoLoader, RepoLoaderConfig


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
