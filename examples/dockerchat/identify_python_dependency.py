import os

DEPENDENCY_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "Pipfile",
    "environment.yml",
    "setup.py",
    "setup.cfg",
]


def identify_dependency_management(directory="."):
    return [
        fname
        for fname in DEPENDENCY_FILES
        if os.path.isfile(os.path.join(directory, fname))
    ]
