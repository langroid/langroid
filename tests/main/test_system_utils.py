from pathlib import Path

import pytest

from langroid.utils.system import (
    create_file,
    diff_files,
    read_file,
    safe_resolve_path,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_create_file_new(temp_dir):
    file_path = temp_dir / "new_file.txt"
    content = "Hello, World!"
    create_file(file_path, content)
    assert file_path.exists()
    assert file_path.read_text() == content


def test_create_file_overwrite(temp_dir):
    file_path = temp_dir / "existing_file.txt"
    original_content = "Original content"
    file_path.write_text(original_content)

    new_content = "New content"
    create_file(file_path, new_content, if_exists="overwrite")
    assert file_path.read_text() == new_content


def test_create_file_skip(temp_dir):
    file_path = temp_dir / "skip_file.txt"
    original_content = "Original content"
    file_path.write_text(original_content)

    new_content = "New content"
    create_file(file_path, new_content, if_exists="skip")
    assert file_path.read_text() == original_content


def test_create_file_error(temp_dir):
    file_path = temp_dir / "error_file.txt"
    file_path.write_text("Existing content")

    with pytest.raises(FileExistsError):
        create_file(file_path, "New content", if_exists="error")


def test_create_file_append(temp_dir):
    file_path = temp_dir / "append_file.txt"
    original_content = "Original content\n"
    file_path.write_text(original_content)

    additional_content = "Additional content"
    create_file(file_path, additional_content, if_exists="append")
    assert file_path.read_text() == original_content + additional_content


def test_create_empty_file(temp_dir):
    file_path = temp_dir / "empty_file.txt"
    create_file(file_path)
    assert file_path.exists()
    assert file_path.read_text() == ""


def test_create_file_in_new_directory(temp_dir):
    new_dir = temp_dir / "new_dir"
    file_path = new_dir / "file_in_new_dir.txt"
    content = "Content in new directory"
    create_file(file_path, content)
    assert file_path.exists()
    assert file_path.read_text() == content


def test_create_file_with_path_object(temp_dir):
    file_path = Path(temp_dir) / "path_object_file.txt"
    content = "Content using Path object"
    create_file(file_path, content)
    assert file_path.exists()
    assert file_path.read_text() == content


def test_read_file(tmp_path):
    file_path = tmp_path / "read_test.txt"
    content = "Line 1\nLine 2\nLine 3"
    file_path.write_text(content)
    assert read_file(str(file_path)) == content


def test_read_file_with_line_numbers(tmp_path):
    file_path = tmp_path / "read_test_numbered.txt"
    content = "Line 1\nLine 2\nLine 3"
    file_path.write_text(content)
    expected = "1: Line 1\n2: Line 2\n3: Line 3"
    assert read_file(str(file_path), line_numbers=True) == expected


def test_read_file_tilde_expansion(tmp_path, monkeypatch):
    # simulate a home directory and read a file via a "~/..." path;
    # read_file must expand "~" before checking existence (regression
    # for a FileNotFoundError raised on a path that actually exists).
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    content = "Line 1\nLine 2"
    (tmp_path / "tilde_read_test.txt").write_text(content)
    assert read_file("~/tilde_read_test.txt") == content


def test_read_file_missing_tilde_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    with pytest.raises(FileNotFoundError):
        read_file("~/no_such_file.txt")


def test_safe_resolve_path_rejects_tilde(tmp_path, monkeypatch):
    # "~" must not be treated as a literal directory name under base_dir:
    # read_file expands it, so the guard has to expand it too.
    home = tmp_path / "home"
    home.mkdir()
    (home / "secret.txt").write_text("SECRET")
    base = tmp_path / "sandbox"
    base.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
    with pytest.raises(ValueError, match="outside the allowed directory"):
        safe_resolve_path(base, "~/secret.txt")


def test_read_file_unresolvable_tilde_user(tmp_path, monkeypatch):
    # "~nosuchuser/..." cannot be expanded; Path.expanduser() raises
    # RuntimeError for it. read_file must still report it as missing, as it
    # documents, rather than propagating RuntimeError.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_file("~nosuchuser12345/f.txt")


def test_safe_resolve_path_unresolvable_tilde_user(tmp_path):
    # an unexpandable "~user" path is treated literally, so it stays inside
    # base_dir and must not raise RuntimeError out of the guard.
    base = tmp_path / "sandbox"
    base.mkdir()
    resolved = safe_resolve_path(base, "~nosuchuser12345/f.txt")
    assert base in resolved.parents


def test_safe_resolve_path_rejects_literal_tilde_symlink_escape(tmp_path, monkeypatch):
    # The tools validate with safe_resolve_path but then operate on the RAW
    # path: create_file/list_dir do not expand "~". So when base_dir is the
    # home dir, "~/x" expands to an in-base path while the literal "base/~/x"
    # can follow a "~" symlink out of the sandbox. Both readings must be
    # checked.
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
    try:
        (home / "~").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        safe_resolve_path(home, "~/secret.txt")


def test_safe_resolve_path_allows_path_within_base(tmp_path):
    base = tmp_path / "sandbox"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "ok.txt").write_text("ok")
    assert safe_resolve_path(base, "sub/ok.txt") == (base / "sub" / "ok.txt").resolve()


def test_diff_files(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("Line 1\nLine 2\nLine 3")
    file2.write_text("Line 1\nLine 2 modified\nLine 3\nLine 4")
    diff = diff_files(str(file1), str(file2))
    assert "Line 2 modified" in diff
    assert "+Line 4" in diff
