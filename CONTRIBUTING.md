# Contributing to Langroid

Thank you for your interest in Langroid! Questions and discussion are welcome on
[Discord](https://discord.gg/ZU36McDgDs).

# Pull requests are by invitation

We accept pull requests only for changes motivated by actual use of Langroid in
your organization, and only after we have agreed on the change in an issue. Pull
requests that were not invited will be closed without review.

**What counts as a use case.** Describe the real application you are building or
running on Langroid: what it does, who uses it, and why it needs this change. If
you are an independent developer, say so, and describe your application the same
way. "Langroid users would want this" is not a use case, and neither is a design
proposal on its own. The larger the change, the higher the bar.

## How to propose a change

1. [Open an issue](https://github.com/langroid/langroid/issues/new/choose) with the
   feature-request or bug-report form. Both ask for your organization name and
   the real application involved, so we can see the need behind the change.
2. We read it and reply on the issue. If the change fits, either we implement it
   ourselves, or we invite you to send the pull request.

Bug reports are held to the same rule: tell us which organization and application
are affected, and how the bug impacts them.

## If we invite your pull request

There are two ways in:

- **From your fork.** Push your work to a branch on your fork of Langroid and
  post the branch name on the issue. We open the pull request from that branch.
  Anything you push to it afterwards updates the pull request.
- **Write access, for regular contributors.** People who contribute regularly
  may be given write access, so they can open pull requests directly. `main`
  is protected, so their changes still go through a reviewed pull request.

Either way, you keep the credit. Your commits stay authored by you, and when a
pull request is squash-merged, GitHub lists you as a co-author of the commit, so
it counts in your GitHub contributions.

## Custom work

If you need Langroid features built or adapted for your organization, Prasad
Chalasani is available for custom work (advisory and development). Contact him
at pchalasani at gmail dot com to discuss terms.

# Contribution Guidelines

These are for contributors whose pull request we have invited (see above).

## Set up dev env

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
to manage dependencies, and `python 3.11` for development.

First install `uv`, then create virtual env and install dependencies:

```bash
# clone this repo and cd into repo root
git clone ...
cd <repo_root>
# create a virtual env under project root, .venv directory
uv venv --python 3.11

# activate the virtual env
. .venv/bin/activate


# use uv to install dependencies (these go into .venv dir)
uv sync --dev 
```

Important note about dependencies management:
> As of version 0.33.0, we are starting to include the `uv.lock` file as part of 
> the repo. This ensures that all contributors are using the same versions of 
> dependencies. If you add a new dependency, `uv add` will automatically update 
> the `uv.lock` file. This will also update the `pyproject.toml` file.

To add packages, use `uv add <package-name>`. This will automatically
find the latest compatible version of the package and add it to `pyproject.
toml`. _Do not manually edit `pyproject.toml` to add packages._

## Set up environment variables (API keys, etc)

Copy the `.env-template` file to a new file `.env` and
insert secrets such as API keys, etc:
- OpenAI API key, Anthropic API key, etc.
- [Optional] GitHub Personal Access Token (needed by  PyGithub to analyze git repos;
  token-based API calls are less rate-limited).
- [Optional] Cache Configs
  - Redis : Password, Host, Port <br>
- Qdrant API key for the vector database.

```bash
cp .env-template .env
# now edit the .env file, insert your secrets as above
``` 

## Run tests
To verify your env is correctly setup, run all tests using `make tests`.

### Running a subset of tests

The full suite is large (and takes ~1.5h in CI), so while iterating you
usually want to run only the tests relevant to your change.

**Locally**, pass file paths, a keyword filter (`-k`), or a marker (`-m`)
to pytest:

```bash
# a single file
pytest -xvs tests/main/test_table_chat_agent.py

# several files
pytest tests/main/test_vector_stores.py tests/main/test_retriever_agent.py

# by keyword (matches test names, params, and file names)
pytest tests/main/test_vector_stores.py -k "qdrant and not hybrid"
```

`-xvs` = exit on first failure, verbose, show output; `-nc` disables the
LLM-response cache; `-n auto` runs tests in parallel via `pytest-xdist`.

**In CI**, instead of waiting for the full `Pytest` workflow, trigger the
manual `pytest-subset` workflow, which runs whatever you pass as
`pytest_args` (it defaults to the Qdrant-related tests):

```bash
# default subset, against a branch
gh workflow run pytest-subset.yml --ref my-branch

# a custom selection
gh workflow run pytest-subset.yml --ref my-branch \
    -f pytest_args="tests/main/test_vector_stores.py -k qdrant_cloud"
```

You can also run it from the GitHub **Actions** tab via "Run workflow".
Note: `workflow_dispatch` only becomes available once the workflow is on
the default branch (`main`).

## IMPORTANT: Please include tests, docs and possibly examples.

For any new features, please include:
- Tests in the `tests` directory (first check if there is a suitable test file to add to).
  _If fixing a bug, please add a regression test, i.e., 
   one which would have failed without your fix_
- A note in `docs/notes` folder, e.g. `docs/notes/weaviate.md` that is a
  (relatively) self-contained guide to using the feature, including any instructions
  on how to set up the environment or keys if needed. 
  See the [weaviate](https://langroid.github.io/langroid/notes/weaviate/) note as an example. Make sure you link to this note in the `mkdocs.yml` file under the `nav` section.
- Where possible and meaningful, add a simple example in the `examples` directory.

## Generate docs

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
and type-checker `mypy`. It also installs a pre-commit hook, 
so that commits are blocked if there are style/type issues. The linting attempts to
auto-fix issues, and warns about those it can't fix.
(There is a separate `make lint` you could do, but that is already part of `make check`).
The `make check` command also looks through the codebase to see if there are any
direct imports from pydantic, and replaces them with importing from `langroid.pydantic_v1`
(this is needed to enable dual-compatibility with Pydantic v1 and v2).

So, typically when submitting a PR, you would do this sequence:
- run `make tests` or `pytest -xvs tests/main/my-specific-test.py` 
  - if needed use `-nc` means "no cache", i.e. to prevent using cached LLM API call responses
  - the `-xvs` option means "exit on first failure, verbose, show output"
- fix things so tests pass, then proceed to lint/style/type checks below.
- `make check` to see what issues there are (typically lints and mypy)
- manually fix any lint or type issues
- `make check` again to see what issues remain
- repeat if needed, until all clean.

When done with these, commit and push to GitHub. If you work from a fork, post
the branch name on the issue and we will open the PR from it; if you have write
access, open the PR yourself. Either way, pushing to the same branch again
updates the PR.

It is strongly recommended to use the `gh` command-line utility when working with git.
Read more in the [GitHub CLI manual](https://cli.github.com/manual/).

## Releasing (maintainers)

A release consists of a version bump + GitHub release, followed by a
PyPI upload. Both steps run non-interactively:

```bash
make all-patch   # or all-minor / all-major
make publish
```

- `make all-patch` bumps the version (via commitizen), pushes `main`
  and the new tag, creates the GitHub release with auto-generated
  notes (`gh release create ... --generate-notes`), and builds the
  wheel + sdist into `dist/`. Release notes can be refined afterwards
  with `gh release edit <version> --notes-file <file>`.
- `make publish` uploads `dist/` to PyPI. It reads `PYPI_TOKEN` from
  the `.env` file in the repository's primary checkout at runtime, so
  the token never appears on a command line or in any output. Add a
  line like this to `.env` (which is gitignored):

  ```bash
  PYPI_TOKEN=pypi-...
  ```

  The target fails with a clear error if `dist/` is missing either
  distribution, if the `.env` file is absent, or if `PYPI_TOKEN` is
  not defined in it. Any `PYPI_TOKEN` already exported in the
  environment is deliberately ignored; the `.env` file is the single
  source of truth.
- Verify the upload independently, e.g.:

  ```bash
  curl -s https://pypi.org/pypi/langroid/json | jq -r .info.version
  ```
