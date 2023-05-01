# llmagent

[![Pytest](https://github.com/langroid/llmagent/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/pytest.yml)
[![Lint](https://github.com/langroid/llmagent/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/llmagent/actions/workflows/mkdocs-deploy.yml)

### Set up dev env

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

### Pull requests

To check for issues locally, run `make check`, it runs linters `black`, 
`flake8` and type-checker `mypy`. The `mypy` flags lots of issues, but 
ignore those for now. 


### Run some examples

1. "Chat" with a set of URLs. 
```bash
python3 examples/urlqa/chat.py
```

To see more output, run with `debug=True`:
```bash
python3 examples/urlqa/chat.py debug=True
```

Ask a question you want answered based on the URLs content. The default 
URLs are about various articles and discussions on LLM-based agents, 
compression and intelligence. If you are using the default URLs, try asking:
> who is Pattie Maes?
and then a follow-up question:
> what did she build?



