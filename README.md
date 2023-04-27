# llmagent

<!--
Fix these badge links later

[![Documentation](https://readthedocs.org/projects/project-name/badge/)](https://project-name.readthedocs.io/)

[![Build Status](https://github.com/username/repository-name/actions/workflows/workflow-name.yml/badge.svg)](https://github.com/username/repository-name/actions)

[![codecov](https://codecov.io/gh/username/repository-name/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repository-name)

[![License](https://img.shields.io/github/license/username/repository-name)](https://github.com/username/repository-name/blob/main/LICENSE)

-->

### Set up dev env

First install `poetry`, then create virtual env and install dependencies:

```bash
# clone the repo
git clone ...

# create a virtual env
python3 -m venv .venv

# activate the virtual env
. .venv/bin/activate

# use poetry to install dependencies
poetry install
```
Copy the `.env-template` file to `.env` and insert your OpenAI API key.
Currently only OpenAI models are supported. Others will be added later.



### Run some examples

1. "Chat" with a set of URLs. 
```bash
python3 examples/urlqa/chatter.py
```

To see more output, run with `debug=True`:
```bash
python3 examples/urlqa/chatter.py debug=True
```

Ask a question you want answered based on the URLs content. The default 
URLs are about various articles and discussions on LLM-based agents, 
compression and intelligence. Try asking:
> who is Pattie Maes?





