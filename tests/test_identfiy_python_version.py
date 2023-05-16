import llmagent.parsing.identify_python_version as pyver

import os
import pytest
import toml
import tempfile
import json

test_version = ">=3.7"


@pytest.fixture
def setup_files():
    files_content = {
        "pyproject.toml": {
            "build-system": {"requires": ["python" + test_version]},
            "tool": {
                "mypy": {"python_version": test_version},
                "poetry": {"dependencies": {"python": test_version}},
            },
            "project": {"requires-python": test_version},
        },
        "requirements.txt": "python" + test_version,
        "runtime.txt": "python" + test_version,
        "setup.cfg": {"options": {"python_requires": test_version}},
        "setup.py": f"""
import setuptools

setuptools.setup(
    python_requires='{test_version}',
)
""",
        "Pipfile": {"requires": {"python_version": test_version}},
        "Pipfile.lock": {"_meta": {"requires": {"python_version": test_version}}},
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files_content.items():
            file_path = os.path.join(temp_dir, filename)
            if (
                filename.endswith(".toml")
                or filename.endswith(".cfg")
                or filename == "Pipfile"
            ):
                with open(file_path, "w") as f:
                    toml.dump(content, f)
            elif filename.endswith(".lock"):
                with open(file_path, "w") as f:
                    json.dump(content, f)
            else:
                with open(file_path, "w") as f:
                    if isinstance(content, str):
                        f.write(content)
                    else:
                        f.write(str(content))
        yield temp_dir


def test_get_python_version_from_pyproject(setup_files):
    starter_response = "According to pyproject.toml "
    os.chdir(setup_files)
    expected_result_build_system = (
        starter_response + "build-system requires python" + test_version
    )
    expected_result_mypy = (
        starter_response + "tool.mypy python_version is " + test_version
    )
    expected_result_poetry = (
        starter_response + "tool.poetry.dependencies python is " + test_version
    )
    expected_result_project = (
        starter_response + "project requires-python is " + test_version
    )
    result = pyver.get_python_version_from_pyproject()
    assert result in [
        expected_result_build_system,
        expected_result_mypy,
        expected_result_poetry,
        expected_result_project,
    ]


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
        == "According to setup.cfg python " + test_version
    )


def test_get_python_version_from_setup_py(setup_files):
    assert (
        pyver.get_python_version_from_setup_py(setup_files)
        == "According to setup.py python" + test_version
    )


def test_get_python_version_from_pipfile(setup_files):
    assert (
        pyver.get_python_version_from_pipfile(setup_files)
        == "According to Pipfile python" + test_version
    )


def test_get_python_version_from_pipfile_lock(setup_files):
    assert (
        pyver.get_python_version_from_pipfile_lock(setup_files)
        == "According to Pipfile.lock python" + test_version
    )
