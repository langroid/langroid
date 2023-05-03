from llmagent.parsing.repo_loader import RepoLoader


def test_repo_loader() -> None:
    """
    Test the RepoLoader class.
    """
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(url)
    docs = repo_loader.load(10)
    assert len(docs) > 0
