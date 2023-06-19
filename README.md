# llmagent

[![Pytest](https://github.com/langroid/llmagent/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/pytest.yml)
[![Lint](https://github.com/langroid/llmagent/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml)

<div align="center">
  <img src="./docs/logos/pure-lambda-non-circular.png" width="100">
</div>

## Set up dev env

We use [`poetry`](https://python-poetry.org/docs/#installation) 
to manage dependencies, and `python 3.11` for development.

First install `poetry`, then create virtual env and install dependencies:

```bash
# clone the repo and cd into repo root
git clone https://github.com/langroid/llmagent.git
cd llmagent

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

## Set up environment variables (API keys, etc

Copy the `.env-template` file to a new file `.env` and 
insert these secrets:
- OpenAI API key, 
- GitHub Personal Access Token (needed by  PyGithub to analyze git repos; 
  token-based API calls are less rate-limited).
- Redis Password (ask @pchalasani for this) for the redis cache.
- Qdrant API key (ask @pchalasani for this) for the vector db.

```bash
cp .env-template .env
# now edit the .env file, insert your secrets as above
``` 

Currently only OpenAI models are supported. Others will be added later.

## Run tests
To verify your env is correctly setup, run all tests using `make test`.

## Generate docs (private only for now)

Generate docs: `make docs`, then go to the IP address shown at the end, like 
`http://127.0.0.1:8000/`
Note this runs a docs server in the background.
To stop it, run `make nodocs`. Also, running `make docs` next time will kill 
any previously running `mkdocs` server.


## Contributing, and Pull requests

In this Python repository, we prioritize code readability and maintainability.
To ensure this, please adhere to the following guidelines when contributing:

1. **Type-Annotate Code:** Add type annotations to function signatures and
   variables to make the code more self-explanatory and to help catch potential
   issues early. For example, `def greet(name: str) -> str:`.

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

To check for issues locally, run `make check`, it runs linters `black`, `ruff`,
`flake8` and type-checker `mypy`. Issues flagged by `black` can usually be 
auto-fixed using `black .`, and to fix `ruff issues`, do:
```
poetry run ruff . --fix
```

- When you run this, `black` may warn that some files _would_ be reformatted. 
If so, you should just run `black .` to reformat them. Also,
- `flake8` may warn about some issues; read about each one and fix those 
  issues.

When done with these, git-commit, push to github and submit the PR. If this 
is an ongoing PR, just push to github again and the PR will be updated. 

Strongly recommend to use the `gh` command-line utility when working with git.
Read more [here](docs/development/github-cli.md).



## Run some examples

### "Chat" with a set of URLs.

```bash
python3 examples/urlqa/chat.py
```

To see more output, run with `--debug` or `-d`:
```bash
python3 examples/urlqa/chat.py -d
```

Ask a question you want answered based on the URLs content. The default 
URLs are about various articles and discussions on LLM-based agents, 
compression and intelligence. If you are using the default URLs, try asking:

> who is Pattie Maes?

and then a follow-up question:

> what did she build?

### "chat"-based dockerfile creator. 
  
This is just a prelim starting point, 
where we leverage the knowledge, reasoning and planning ability of the LLM.
We don't hard-code any logic. All the smarts are in the initial prompt, 
which instructs the LLM to _ask any info it needs_ to help it build the 
dockerfile. The LLM then generates a series of questions, answered by the 
human. The next step will be to nearly eliminate the human from this loop, 
and have the LLM questions trigger scripts or semantic lookups in the 
sharded + vectorized code-repo. The LLM could then show the answer it found 
or give a set of possible options, and ask the human to confirm/choose.  


```bash
python3 examples/dockerfile/chat.py
```

By default this uses `gpt-3.5-turbo`. 
For better results, you can specify the option `-4`, so it uses `gpt4` 
(CAUTION GPT4 is ~20-30 times more expensive per token):
```
python3 examples/dockerfile/chat.py -4
```

