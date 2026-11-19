"""RepoLoader must never let a repository URL be parsed as a git option."""

import subprocess
from pathlib import Path

import pytest

from langroid.parsing.repo_loader import RepoLoader


@pytest.mark.parametrize("url", ["--bare", " -q github.com/langroid/langroid"])
def test_option_like_url_is_rejected(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        RepoLoader(url)


def test_clone_ends_option_parsing_before_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even if `url` starts with "-", clone treats it as a repository.

    Without `--`, `git clone --bare <src>` would succeed (a bare clone of the
    local repo `src`); with it, git looks for a repository named "--bare" and
    fails, so `clone()` returns None.
    """
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run(
        ["git", "-C", str(src), "commit", "-q", "--allow-empty", "-m", "init"],
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(Path.home()),
            "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        },
    )

    loader = RepoLoader.__new__(RepoLoader)  # skip the GitHub API lookup
    loader.url = "--bare"
    loader.clone_path = None
    loader.log_file = str(tmp_path / "download_log.json")
    (tmp_path / "download_log.json").write_text("{}")

    assert loader.clone(str(src)) is None
    assert not (tmp_path / "src.git").exists()
