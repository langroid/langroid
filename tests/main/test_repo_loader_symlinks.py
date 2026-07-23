"""Tests that ``RepoLoader`` does not follow symlinks out of the ingested root.

Regression tests for GHSA-99m6-3pvg-q66h: a repository or folder containing
symlinks pointing outside the root caused ``load_from_folder`` and
``get_documents`` to read and expose host files as ``Document`` objects.

These are pure filesystem tests -- real directories, real symlinks, no mocks.
Every escape test also asserts that an ordinary in-root file *was* loaded, so
a test cannot pass vacuously by returning no documents at all.
"""

from pathlib import Path
from typing import Callable, List

import pytest

from langroid.mytypes import Document
from langroid.parsing.repo_loader import RepoLoader

SECRET = "HOST-SECRET-SHOULD-NOT-BE-READ"
NESTED_SECRET = "NESTED-SECRET-SHOULD-NOT-BE-READ"
INSIDE = "INSIDE-CONTENT-IS-FINE"

Loader = Callable[[Path], List[Document]]


@pytest.fixture(params=["load_from_folder", "get_documents"])
def loader(request: pytest.FixtureRequest) -> Loader:
    """Run either local-tree ingestion entry point, behind one signature."""
    if request.param == "load_from_folder":

        def load(path: Path) -> List[Document]:
            return RepoLoader.load_from_folder(
                str(path),
                depth=5,
                lines=20,
                file_types=["txt"],
            )[1]

        return load

    def get(path: Path) -> List[Document]:
        return RepoLoader.get_documents(str(path), depth=5, file_types=["txt"])

    return get


def _all_text(docs: List[Document]) -> str:
    return "\n".join(d.content for d in docs)


def test_escaping_file_and_dir_symlinks_are_skipped(
    loader: Loader, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")
    (outside / "nested.txt").write_text(NESTED_SECRET, encoding="utf-8")
    (base / "ok.txt").write_text(INSIDE, encoding="utf-8")
    (base / "link-file.txt").symlink_to(outside / "secret.txt")
    (base / "lootdir").symlink_to(outside, target_is_directory=True)

    text = _all_text(loader(base))
    assert INSIDE in text
    assert SECRET not in text
    assert NESTED_SECRET not in text


def test_relative_escaping_symlinks_are_skipped(loader: Loader, tmp_path: Path) -> None:
    """Escapes spelled with ``..`` rather than an absolute target."""
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = base / "one" / "two"
    nested.mkdir(parents=True)
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    (outside / "nested.txt").write_text(NESTED_SECRET, encoding="utf-8")
    (base / "ok.txt").write_text(INSIDE, encoding="utf-8")
    (nested / "loot.txt").symlink_to("../../../secret.txt")
    (base / "lootdir").symlink_to("../outside", target_is_directory=True)

    text = _all_text(loader(base))
    assert INSIDE in text
    assert SECRET not in text
    assert NESTED_SECRET not in text


def test_in_root_symlinks_are_still_followed(loader: Loader, tmp_path: Path) -> None:
    """Only escaping symlinks are skipped; in-root ones keep working."""
    base = tmp_path / "base"
    nested = base / "one"
    nested.mkdir(parents=True)
    (nested / "real.txt").write_text(INSIDE, encoding="utf-8")
    (base / "absolute.txt").symlink_to(nested / "real.txt")
    (base / "relative.txt").symlink_to("one/real.txt")

    # Reached via the real path and via both in-root symlinks.
    assert _all_text(loader(base)).count(INSIDE) == 3


def test_symlinked_ingestion_root_is_supported(loader: Loader, tmp_path: Path) -> None:
    """A symlink passed *as* the root must not make everything look external."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "ok.txt").write_text(INSIDE, encoding="utf-8")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    assert INSIDE in _all_text(loader(linked_root))


def _broken(base: Path) -> None:
    (base / "broken.txt").symlink_to(base / "missing.txt")


def _self_referential(base: Path) -> None:
    (base / "self.txt").symlink_to("self.txt")


def _mutual_loop(base: Path) -> None:
    (base / "first.txt").symlink_to("second.txt")
    (base / "second.txt").symlink_to("first.txt")


@pytest.mark.parametrize(
    "make_bad",
    [_broken, _self_referential, _mutual_loop],
    ids=["broken", "self-referential", "mutual-loop"],
)
def test_unresolvable_symlinks_are_skipped_without_aborting(
    loader: Loader, tmp_path: Path, make_bad: Callable[[Path], None]
) -> None:
    """A bad link skips that entry; it must not abort the whole ingestion.

    ``Path.resolve()`` raises ``RuntimeError`` -- not ``OSError`` -- on a
    symlink loop, which otherwise propagates out of the walk and kills the run.
    """
    base = tmp_path / "base"
    base.mkdir()
    (base / "ok.txt").write_text(INSIDE, encoding="utf-8")
    make_bad(base)

    assert INSIDE in _all_text(loader(base))
