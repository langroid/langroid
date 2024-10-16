from pathlib import Path

import pytest

from langroid.utils.system import create_file, diff_files, read_file


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


def test_diff_files(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("Line 1\nLine 2\nLine 3")
    file2.write_text("Line 1\nLine 2 modified\nLine 3\nLine 4")
    diff = diff_files(str(file1), str(file2))
    assert "Line 2 modified" in diff
    assert "+Line 4" in diff
