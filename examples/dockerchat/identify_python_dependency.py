import os


def identify_dependency_management(directory="."):
    req_file_path = os.path.join(directory, "requirements.txt")
    toml_file_path = os.path.join(directory, "pyproject.toml")
    pipfile_path = os.path.join(directory, "Pipfile")
    pipfile_lock_path = os.path.join(directory, "Pipfile.lock")
    conda_env_path = os.path.join(directory, "environment.yml")
    setup_py_path = os.path.join(directory, "setup.py")
    setup_cfg_path = os.path.join(directory, "setup.cfg")

    dependency_management = []

    if os.path.exists(req_file_path):
        dependency_management.append("requirements.txt")

    if os.path.exists(toml_file_path):
        dependency_management.append("pyproject.toml (Poetry)")

    elif os.path.exists(pipfile_path) or os.path.exists(pipfile_lock_path):
        dependency_management.append("Pipfile (Pipenv)")

    if os.path.exists(conda_env_path):
        dependency_management.append("environment.yml (Conda)")

    if os.path.exists(setup_py_path):
        dependency_management.append("setup.py")

    if os.path.exists(setup_cfg_path):
        dependency_management.append("setup.cfg")

    return dependency_management

    # if dependency_management:
    #     return f"Dependencies in this repo are managed using: {dependency_management}"
    # else:
    #     return "Dependencies are not defined in this repo"
