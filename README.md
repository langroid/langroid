<div style="display: flex; align-items: center;">
  <img src="docs/assets/orange-logo.png" alt="Logo" 
        width="80" height="80"align="left">
  <h1>Langroid</h1>
</div>

[![Pytest](https://github.com/langroid/langroid/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/pytest.yml)
[![Lint](https://github.com/langroid/langroid/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml)


## Contributors:
- Prasad Chalasani (Independent ML Consultant)
- Somesh Jha (Professor of CS, U Wisc at Madison)
- Mohannad Alhanahnah (Research Associate, U Wisc at Madison)
- Ashish Hooda (PhD Candidate, U Wisc at Madison)

## Set up dev env

We use [`poetry`](https://python-poetry.org/docs/#installation) 
to manage dependencies, and `python 3.11` for development.

First install `poetry`, then create virtual env and install dependencies:

```bash
# clone the repo and cd into repo root
git clone https://github.com/langroid/langroid.git
cd langroid

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

You can also run `make lint` to (try to) auto-tix `black` and `ruff`
issues. 

So, typically when submitting a PR, you would do this sequence:
- run `pytest tests -nc` (`-nc` means "no cache", i.e. do not use cached LLM 
  API call responses)
- fix things so tests pass, then proceed to lint/style/type checks
- `make check` to see what issues there are
- `make lint` to auto-fix some of them
- `make check` again to see what issues remain
- possibly manually fix `flake8` issues, and any `mypy` issues flagged.
- `make check` again to see if all issues are fixed.
- repeat if needed, until all clean. 

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

### "Chat" with a code repo, given the GitHub URL
```bash
python3 examples/codechat/codechat.py
```

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

By default this uses `gpt-4-0613`.



## Logs of multi-agent interactions

When running a multi-agent chat, e.g. using `task.run()`, two types of logs 
are generated:
- plain-text logs in `logs/<task_name>.log`
- tsv logs in `logs/<task_name>.tsv`

We will go into details of inter-agent chat structure in another place, 
but for now it is important to realize that the logs show _every attempt at 
  responding to the current pending message, even those that are not allowed_.
The ones marked with an asterisk (*) are the ones that are considered the 
responses for a given `step()` (which is a "turn" in the conversation).

The plain text logs have color-coding ANSI chars to make them easier to read 
by doing `less <log_file>`. The format is:
```
(TaskName) Responder SenderEntity (EntityName) (=> Recipient) TOOL Content
```

The structure of the `tsv` logs is similar. A great way to view these is to 
install and use `visidata` (https://www.visidata.org/):
```bash
vd logs/<task_name>.tsv
```

