"""Tests that ``RepoLoader`` does not follow symlinks out of the ingested root.

Regression tests for GHSA-99m6-3pvg-q66h: a repository or folder containing
symlinks pointing outside the root caused ``load_from_folder`` and
``get_documents`` to read and expose host files as ``Document`` objects.

These are pure filesystem tests -- real directories, real symlinks, no mocks.
"""

from pathlib import Path

import pytest

from langroid.parsing.repo_loader import RepoLoader

SECRET = "HOST-SECRET-SHOULD-NOT-BE-READ"
NESTED_SECRET = "NESTED-SECRET-SHOULD-NOT-BE-READ"
INSIDE = "INSIDE-CONTENT-IS-FINE"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Build ``base/`` with symlinks escaping to a sibling ``outside/``."""
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()

    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")
    (outside / "nested.txt").write_text(NESTED_SECRET, encoding="utf-8")
    (base / "ok.txt").write_text(INSIDE, encoding="utf-8")

    (base / "link-file.txt").symlink_to(outside / "secret.txt")
    (base / "lootdir").symlink_to(outside, target_is_directory=True)
    return base


def _all_text(docs) -> str:
    return "\n".join(d.content for d in docs)


def test_load_from_folder_does_not_follow_escaping_symlinks(tree: Path) -> None:
    _structure, docs = RepoLoader.load_from_folder(
        str(tree),
        depth=3,
        lines=20,
        file_types=["txt"],
    )
    text = _all_text(docs)
    assert INSIDE in text
    assert SECRET not in text
    assert NESTED_SECRET not in text


def test_get_documents_does_not_follow_escaping_symlinks(tree: Path) -> None:
    docs = RepoLoader.get_documents(str(tree), depth=3, file_types=["txt"])
    text = _all_text(docs)
    assert INSIDE in text
    assert SECRET not in text
    assert NESTED_SECRET not in text


def test_symlinks_pointing_inside_the_root_are_still_followed(
    tmp_path: Path,
) -> None:
    """Only escaping symlinks are skipped; in-root symlinks keep working."""
    base = tmp_path / "base"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "real.txt").write_text(INSIDE, encoding="utf-8")
    (base / "alias.txt").symlink_to(base / "sub" / "real.txt")

    docs = RepoLoader.get_documents(str(base), depth=3, file_types=["txt"])
    # Reached both via the real path and via the in-root symlink.
    assert sum(INSIDE in d.content for d in docs) == 2
