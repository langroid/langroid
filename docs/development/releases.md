!!! note
    By GPT4. Caveat Lector. May not be fully accurate. Trust but Verify!

Setting up releases and versioning for a project hosted on GitHub can be
streamlined using tools like Poetry, which already manages dependencies and
packaging for Python projects. To set up releases and versioning for your
project, follow these steps:

1. Install Poetry:
   If you haven't already, install Poetry by running the following command in
   your terminal:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Initialize your project with Poetry:
   In your project's root directory, run the following command to create
   a `pyproject.toml` file:

```bash
poetry init
```

Follow the prompts to configure your project's details. This will generate
a `pyproject.toml` file containing your project's metadata and dependencies.

3. Specify the version in `pyproject.toml`:
   Poetry uses [Semantic Versioning](https://semver.org/). You can set the
   version of your project in the `pyproject.toml` file. For example:

```toml
[tool.poetry]
name = "your_project_name"
version = "0.1.0"
description = "Your project description"
authors = ["Your Name <you@example.com>"]
```

4. Configure versioning with Git:
   Start by initializing Git in your project directory (if you haven't already)
   and commit the initial changes:

```bash
git init
git add .
git commit -m "Initial commit"
```

Create a `.gitignore` file in your project's root directory to exclude files and
directories that should not be tracked by Git. For a Python project, you might
want to ignore:

```
__pycache__/
*.pyc
*.pyo
*.pyd
*.pyc
.venv/
dist/
build/
*.egg-info/
*.egg
```

5. Tagging a release:
   To create a release, you can use Git tags. First, ensure your changes are
   committed, then create a new tag with the following command:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
```

Replace `v0.1.0` with the appropriate version number. You can push the tags to
GitHub by running:

```bash
git push origin --tags
```

6. Automate the release process using GitHub Actions (optional):
   You can use GitHub Actions to automate your release process. Create
   a `.github/workflows/release.yml` file in your project's root directory, and
   add the following:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.x

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -

      - name: Install dependencies
        run: poetry install

      - name: Build package
        run: poetry build

      - name: Publish package
        run: poetry publish
        env:
          PYPI_USERNAME: ${{ secrets.PYPI_USERNAME }}
          PYPI_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
```

This workflow is triggered when you push a new tag. It checks out your code,
sets up Python, installs Poetry, installs dependencies, builds the package, and
publishes it to PyPI.

Make sure to set the `PYPI_USERNAME` and `PYPI_PASSWORD` in your GitHub
repository's secrets to securely store your PyPI credentials.

With this setup, you can now manage releases and versioning for your
GitHub-hosted Python project using Poetry and `pyproject.toml`. Here's a summary
of the steps for future reference:

1. Install Poetry and initialize your project with `poetry init`.
2. Specify your project's version in the `pyproject.toml` file.
3. Configure versioning with Git, create a `.gitignore` file, and commit your
   changes.
4. Use Git tags to create and push releases to GitHub.
5. Optionally, automate the release process using GitHub Actions.

When you want to update your project's version, simply update the version number
in the `pyproject.toml` file, commit the changes, and create a new Git tag. If
you set up the optional GitHub Actions workflow, it will automatically build and
publish your package to PyPI when you push the new tag.

# Bumping release versions

Yes, there are tools to help you automate the version bumping process. One such
tool is `bump2version` (formerly `bumpversion`). Here's how you can use it to
bump your project's version:

1. Install `bump2version`:
   You can install `bump2version` using `pip`:

```bash
pip install bump2version
```

2. Configure `bump2version`:
   Create a `.bumpversion.cfg` file in your project's root directory with the
   following content:

```ini
[bumpversion]
current_version = 0.1.0
commit = True
tag = True
tag_name = v{new_version}

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"
```

Replace `0.1.0` with your project's current version. The configuration file
tells `bump2version` to update the version in `pyproject.toml`, commit the
changes, and create a new Git tag with the updated version number.

3. Bump the version:
   To bump the version, run the following command:

```bash
bump2version <part>
```

Replace `<part>` with `major`, `minor`, or `patch` depending on the type of
version bump you want to perform. This will update the version
in `pyproject.toml`, commit the changes, and create a new Git tag.

For example, to bump the minor version, run:

```bash
bump2version minor
```

This will update the version from `0.1.0` to `0.2.0`.

4. Push the changes and tags to GitHub:
   After bumping the version, push the changes and tags to GitHub:

```bash
git push && git push --tags
```

Now, your project's version is bumped, and the new version is tagged in your
GitHub repository.

When you want to bump the version in the future, simply run the `bump2version`
command with the appropriate `<part>` argument, and it will take care of
updating the version, committing the changes, and creating a new Git tag.