import toml
import configparser
import re
import os


def get_python_version_from_pyproject(directory: str = ".") -> str:
    """
    Inspect the file pyproject.toml in the root directory for a given repo to extract python version whether listed under [build-system], [tool.poetry.dependencies], or [tool.mypy]
    Args:
        directory (str): location of pyproject.toml, by default the root directory
    Returns:
        str: python version
    """
    try:
        with open(os.path.join(directory, "pyproject.toml")) as file:
            content = toml.load(file)

        starter_response = "According to pyproject.toml "
        python_version = content.get("build-system", {}).get("requires", [None])[0]
        if python_version and python_version.startswith("python"):
            return starter_response + python_version
        else:
            # Check under [tool.mypy] section
            mypy_version = content.get("tool", {}).get("mypy", {}).get("python_version")
            if mypy_version:
                return starter_response + "python" + mypy_version
            else:
                # Check under [tool.poetry.dependencies] section
                poetry_version = (
                    content.get("tool", {})
                    .get("poetry", {})
                    .get("dependencies", {})
                    .get("python")
                )
                if poetry_version:
                    return starter_response + "python" + poetry_version
    except FileNotFoundError:
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
            return starter_response + "python" + python_version.strip("'\"")
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
