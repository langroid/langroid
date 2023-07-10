import configparser
import json
import os
import re

import toml


def get_python_version_from_pyproject(directory=".") -> str:
    """
    Inspect the file pyproject.toml in the root directory for a given repo to extract python version whether listed under [build-system], [tool.poetry.dependencies], or [tool.mypy]
    Args:
        directory (str): location of pyproject.toml, by default the root directory
    Returns:
        str: python version
    """
    pyproject_file = os.path.join(directory, "pyproject.toml")

    starter_response = "According to pyproject.toml "

    if os.path.exists(pyproject_file):
        with open(pyproject_file, "r") as f:
            content = toml.load(f)

        if "build-system" in content and "requires" in content["build-system"]:
            for requirement in content["build-system"]["requires"]:
                if requirement.startswith("python"):
                    return starter_response + "build-system requires " + requirement

        if "tool" in content:
            if (
                "mypy" in content["tool"]
                and "python_version" in content["tool"]["mypy"]
            ):
                return (
                    starter_response
                    + "tool.mypy python_version is "
                    + content["tool"]["mypy"]["python_version"]
                )
            if (
                "poetry" in content["tool"]
                and "dependencies" in content["tool"]["poetry"]
            ):
                if "python" in content["tool"]["poetry"]["dependencies"]:
                    return (
                        starter_response
                        + "tool.poetry.dependencies python is "
                        + content["tool"]["poetry"]["dependencies"]["python"]
                    )

        if "project" in content and "requires-python" in content["project"]:
            return (
                starter_response
                + "project requires-python "
                + content["project"]["requires-python"]
            )

    return None


def get_python_version_from_requirements(directory=".") -> str:
    """
    Inspect the file requirements.txt in the root directory for a given repo to extract python version
    Args:
        directory (str): location of requirements.txt, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "requirements.txt")) as file:
            requirements = file.readlines()

        starter_response = "According to requirements.txt "
        python_version = [v for v in requirements if v.startswith("python")]

        if python_version:
            return starter_response + python_version[0].strip()
    except FileNotFoundError:
        return None


def get_python_version_from_runtime(directory=".") -> str:
    """
    Inspect the file runtime.txt in the root directory for a given repo to extract python version
    Args:
        directory (str): location of runtime.txt, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "runtime.txt")) as file:
            runtime = file.readline().strip()

        starter_response = "According to runtime.txt "
        if runtime.startswith("python"):
            return starter_response + runtime
    except FileNotFoundError:
        return None


def get_python_version_from_setup_cfg(directory=".") -> str:
    """
    Inspect the file setup.cfg in the root directory for a given repo to extract python version
    Args:
        directory (str): location of setup.cfg, by default the root directory
    Returns:
        str: python version
    """
    try:
        config = configparser.ConfigParser()
        config.read(os.path.join(directory, "setup.cfg"))
        starter_response = "According to setup.cfg "
        python_version = config.get("options", "python_requires", fallback=None)
        if python_version:
            # Remove any quotes from the start and end of the version string
            return starter_response + "python " + python_version.strip("'\"")
    except FileNotFoundError:
        return None


def get_python_version_from_setup_py(directory=".") -> str:
    """
    Inspect the file setup.py in the root directory for a given repo to extract python version
    This will match lines where python_requires is followed by either = or : and then a string, which will cover both the cases where python_requires is a direct argument to setuptools.setup() or a key in a dictionary
    Args:
        directory (str): location of setup.py, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "setup.py")) as file:
            content = file.read()

        starter_response = "According to setup.py "
        match = re.search(r"python_requires\s*[:=]\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return starter_response + "python" + match.group(1)
    except FileNotFoundError:
        return None


def get_python_version_from_pipfile(directory=".") -> str:
    """
    Inspect the file Pipfile in the root directory for a given repo to extract python version
    Args:
        directory (str): location of Pipfile, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "Pipfile")) as file:
            content = toml.load(file)
        starter_response = "According to Pipfile "
        python_version = content.get("requires", {}).get("python_version")
        if python_version:
            return starter_response + "python" + python_version
    except FileNotFoundError:
        return None


def get_python_version_from_pipfile_lock(directory=".") -> str:
    """
    Inspect the file Pipfile.lock in the root directory for a given repo to extract python version
    Args:
        directory (str): location of Pipfile.lock, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "Pipfile.lock")) as file:
            content = json.load(file)
        starter_response = "According to Pipfile.lock "
        python_version = (
            content.get("_meta", {}).get("requires", {}).get("python_version")
        )
        if python_version:
            return starter_response + "python" + python_version
    except FileNotFoundError:
        return None


def get_python_version(directory="."):
    python_version = (
        get_python_version_from_pyproject(directory)
        or get_python_version_from_requirements(directory)
        or get_python_version_from_runtime(directory)
        or get_python_version_from_setup_cfg(directory)
        or get_python_version_from_setup_py(directory)
        or get_python_version_from_pipfile_lock(directory)
        or get_python_version_from_pipfile(directory)
    )
    return python_version
