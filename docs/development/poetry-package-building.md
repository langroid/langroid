# Building a package from a poetry-based project

!!! note
    Caveat Lector. May not be fully accurate. Trust but Verify!



Great! If you already have a `pyproject.toml` file and a Poetry-based virtual
environment setup, you can follow these steps to create a package that can be
installed by others from GitHub:

1. Make sure your package information is complete in `pyproject.toml`:

Ensure that your `pyproject.toml` contains the necessary package information,
including name, version, description, authors, and dependencies. For example:

```toml
[tool.poetry]
name = "your-package-name"
version = "0.1.0"
description = "A short description of your package"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
# Add other dependencies here

[tool.poetry.dev-dependencies]
# Add development dependencies here
```

2. Build the package:

To build your package, run the following command in your project directory:

```bash
poetry build
```

This will generate a `.tar.gz` source distribution and a `.whl` wheel
distribution in the `dist` folder.

3. Push your code to GitHub:

If you haven't already, create a GitHub repository for your project and push
your code to it. Make sure to include a README file with a brief description of
your package, installation instructions, and usage examples.

4. Install the package directly from GitHub:

Once your project is on GitHub, users can install your package using `pip`. To
do this, they can run the following command, replacing `username` with your
GitHub username and `your-package-name` with the name of your repository:

```bash
pip install git+https://github.com/username/your-package-name.git
```

5. (Optional) Publish your package to PyPI:

If you want to make your package more easily discoverable and installable, you
can publish it to the Python Package Index (PyPI). To do this, first install the
necessary tools:

```bash
pip install --upgrade poetry twine
```

Then, log in to PyPI using:

```bash
poetry config pypi-token.pypi <your_pypi_token>
```

You can find your PyPI token in your account settings
on [pypi.org](https://pypi.org/manage/account/token/).

Finally, publish your package with:

```bash
poetry publish --build
```

Now, users can install your package using just the package name:

```bash
pip install your-package-name
```