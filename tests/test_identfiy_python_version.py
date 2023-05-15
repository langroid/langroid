import llmagent.parsing.identify_python_version as pyver

import os
import pytest
import toml
import tempfile

test_version = ">=3.7"


@pytest.fixture
def setup_files():
    files_content = {
        "pyproject.toml": {"build-system": {"requires": ["python" + test_version]}},
        "requirements.txt": "python" + test_version,
        "runtime.txt": "python" + test_version,
        "setup.cfg": {"options": {"python_requires": test_version}},
        "setup.py": f"""
import setuptools

setuptools.setup(
    python_requires='{test_version}',
)
""",
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files_content.items():
            file_path = os.path.join(temp_dir, filename)
            if filename.endswith(".toml") or filename.endswith(".cfg"):
                with open(file_path, "w") as f:
                    toml.dump(content, f)
            else:
                with open(file_path, "w") as f:
                    f.write(content)
        yield temp_dir


def test_get_python_version_from_pyproject(setup_files):
    assert (
        pyver.get_python_version_from_pyproject(setup_files)
        == "According to pyproject.toml python" + test_version
    )


def test_get_python_version_from_requirements(setup_files):
    assert (
        pyver.get_python_version_from_requirements(setup_files)
        == "According to requirements.txt python" + test_version
    )


def test_get_python_version_from_runtime(setup_files):
    assert (
        pyver.get_python_version_from_runtime(setup_files)
        == "According to runtime.txt python" + test_version
    )


def test_get_python_version_from_setup_cfg(setup_files):
    assert (
        pyver.get_python_version_from_setup_cfg(setup_files)
        == "According to setup.cfg python" + test_version
    )


def test_get_python_version_from_setup_py(setup_files):
    assert (
        pyver.get_python_version_from_setup_py(setup_files)
        == "According to setup.py python" + test_version
    )
