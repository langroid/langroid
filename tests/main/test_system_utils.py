from langroid.utils.system import create_file, diff_files, read_file


def test_create_file(tmp_path):
    file_path = tmp_path / "test_file.txt"
    content = "Test content"
    create_file(file_path, content)
    assert file_path.exists()
    assert file_path.read_text() == content


def test_create_empty_file(tmp_path):
    file_path = tmp_path / "empty_file.txt"
    create_file(file_path)
    assert file_path.exists()
    assert file_path.read_text() == ""


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
