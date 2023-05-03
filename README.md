# llmagent

[![Pytest](https://github.com/langroid/llmagent/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/pytest.yml)
[![Lint](https://github.com/langroid/llmagent/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml)

## Set up dev env

We use [`poetry`](https://python-poetry.org/docs/#installation) 
to manage dependencies, and `python 3.11` for development.

First install `poetry`, then create virtual env and install dependencies:

```bash
# clone the repo and cd into repo root
git clone ...
cd llmagent

# create a virtual env under project root, .venv directory
python3 -m venv .venv

# activate the virtual env
. .venv/bin/activate

# use poetry to install dependencies (these go into .venv dir)
poetry install
```
Copy the `.env-template` file to a new file `.env` and 
insert your OpenAI API key:
```bash
cp .env-template .env
# now edit the .env file, insert your OpenAI API key
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


## Pull requests

To check for issues locally, run `make check`, it runs linters `black`, 
`flake8` and type-checker `mypy`. The `mypy` flags lots of issues, but 
ignore those for now. 

- When you run this, `black` may warn that some files _would_ be reformatted. 
If so, you should just run `black .` to reformat them. Also,
- `flake8` may warn about some issues; read about each one and fix those 
  issues.

When done with these, git-commit, push to github and submit the PR. If this 
is an ongoing PR, just push to github again and the PR will be updated. 

Strongly recommend to use the `gh` command-line utility when working with git.
Read more [here](docs/writeups/github-cli.md).



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

