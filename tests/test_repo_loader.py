from llmagent.parsing.repo_loader import RepoLoader


def test_repo_loader() -> None:
    """
    Test the RepoLoader class.
    """
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(
        url,
        extensions=[
            "py",
            "md",
            "yml",
            "yaml",
            "txt",
            "text",
            "sh",
            "toml",
            "cfg",
            "ini",
        ],
    )
    docs = repo_loader.load(10, depth=0, lines=20)
    assert len(docs) > 0
    assert len(docs) <= 10
    for doc in docs:
        assert len(doc.content.split("\n")) <= 20
