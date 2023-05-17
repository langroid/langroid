from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
import tempfile
import os
import json


def test_repo_loader() -> None:
    """
    Test the RepoLoader class.
    """
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(url, config=RepoLoaderConfig())

    docs = repo_loader.load(10, depth=0, lines=20)
    assert len(docs) > 0
    assert len(docs) <= 10
    for doc in docs:
        assert len(doc.content.split("\n")) <= 20

    # tree structure direct from github
    tree = repo_loader.get_repo_structure(depth=2)
    assert len(tree) > 0

    tree_with_contents = repo_loader.get_repo_structure(depth=1, lines=5)
    assert len(tree_with_contents) > 0

    # list all names to depth 2
    listing = repo_loader.ls(tree_with_contents, depth=2)
    assert len(listing) > 0

    # cloning
    repo_loader.clone()
    assert len(os.listdir(repo_loader.clone_path)) > 0
    # tree structure from cloned repo
    folder_tree = repo_loader.get_folder_structure(depth=2)
    assert len(folder_tree) > 0

    folder_tree_with_contents = repo_loader.get_folder_structure(
        repo_loader.clone_path, depth=2, lines=5
    )
    assert len(folder_tree_with_contents) > 0

    # dump to json
    s = json.dumps(folder_tree_with_contents, indent=2)
    assert len(s) > 0

    # select specific files
    desired = ["workflows", "Makefile", "pyproject.toml"]
    subtree = RepoLoader.select(
        folder_tree_with_contents,
        names=desired,
    )

    assert len(subtree["dirs"]) + len(subtree["files"]) <= 3

    # list all names to depth 2
    listing = repo_loader.ls(folder_tree_with_contents, depth=2)
    assert len(listing) > 0

