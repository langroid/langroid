# Contributing to Langroid

We designed Langroid with a transparent, flexible architecture, and
we welcome contributions from the community. Here are some guidelines.

## Set up dev env

We use [`poetry`](https://python-poetry.org/docs/#installation)
to manage dependencies, and `python 3.11` for development.

First install `poetry`, then create virtual env and install dependencies:

```bash
# clone this repo and cd into repo root
git clone ...
cd <repo_root>
# create a virtual env under project root, .venv directory
python3 -m venv .venv

# activate the virtual env
. .venv/bin/activate

# use poetry to install dependencies (these go into .venv dir)
poetry install

```
To add packages, use `poetry add <package-name>`. This will automatically
find the latest compatible version of the package and add it to `pyproject.
toml`. _Do not manually edit `pyproject.toml` to add packages._

## Set up environment variables (API keys, etc)

Copy the `.env-template` file to a new file `.env` and
insert these secrets:
- OpenAI API key,
- GitHub Personal Access Token (needed by  PyGithub to analyze git repos;
  token-based API calls are less rate-limited).
- Redis Password for the redis cache.
- Qdrant API key for the vector database.

```bash
cp .env-template .env
# now edit the .env file, insert your secrets as above
``` 

Currently only OpenAI models are supported. 
You are welcome to submit a PR to submit other API-based or local models. 

## Run tests
To verify your env is correctly setup, run all tests using `make tests`.

## Generate docs (private only for now)

Generate docs: `make docs`, then go to the IP address shown at the end, like
`http://127.0.0.1:8000/`
Note this runs a docs server in the background.
To stop it, run `make nodocs`. Also, running `make docs` next time will kill
any previously running `mkdocs` server.


## Coding guidelines

In this Python repository, we prioritize code readability and maintainability.
To ensure this, please adhere to the following guidelines when contributing:

1. **Type-Annotate Code:** Add type annotations to function signatures and
   variables to make the code more self-explanatory and to help catch potential
   issues early. For example, `def greet(name: str) -> str:`. We use [`mypy`](https://mypy.readthedocs.io/en/stable/) for
   type-checking, so please ensure your code passes `mypy` checks. 

2. **Google-Style Docstrings:** Use Google-style docstrings to clearly describe
   the purpose, arguments, and return values of functions. For example:

   ```python
   def greet(name: str) -> str:
       """Generate a greeting message.

       Args:
           name (str): The name of the person to greet.

       Returns:
           str: The greeting message.
       """
       return f"Hello, {name}!"
   ```

3. **PEP8-Compliant 80-Char Max per Line:** Follow the PEP8 style guide and keep
   lines to a maximum of 80 characters. This improves readability and ensures
   consistency across the codebase.

If you are using an LLM to write code for you, adding these
instructions will usually get you code compliant with the above:
```
use type-annotations, google-style docstrings, and pep8 compliant max 80 
     chars per line.
```     


By following these practices, we can create a clean, consistent, and
easy-to-understand codebase for all contributors. Thank you for your
cooperation!

## Submitting a PR

To check for issues locally, run `make check`, it runs linters `black`, `ruff`,
`flake8` and type-checker `mypy`. It also installs a pre-commit hook, 
so that commits are blocked if there are style/type issues.
Issues flagged by `black` or `ruff` can usually be fixed by running `make lint`. 
`flake8` may warn about some issues; read about each one and fix those
  issues.

So, typically when submitting a PR, you would do this sequence:
- run `make tests` or `pytest tests/main` (if needed use `-nc` means "no cache", i.e. to prevent
  using cached LLM API call responses)
- fix things so tests pass, then proceed to lint/style/type checks below.
- `make check` to see what issues there are
- `make lint` to auto-fix some of them
- `make check` again to see what issues remain
- possibly manually fix `flake8` issues, and any `mypy` issues flagged.
- `make check` again to see if all issues are fixed.
- repeat if needed, until all clean.

When done with these, commit and push to github and submit the PR. If this
is an ongoing PR, just push to github again and the PR will be updated.

Strongly recommend to use the `gh` command-line utility when working with git.
Read more [here](docs/development/github-cli.md).
