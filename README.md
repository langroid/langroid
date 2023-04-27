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

```bash

### Run some examples

1. "Chat" with a set of URLs

This is just a very simple script, for myself to understand the flow, to see 
what issues and difficulties I run into, and to inspire ideas. This uses 
langchain's `ConversationalRetrievalChain` chain.

"Ask a question" you want answered based on the URLs content. The default 
URLs are about CMU financial aid. You can try asking:
> what is total undergrad cost?


```bash
python3 examples/urlqa/chat-flat.py debug=False
```



