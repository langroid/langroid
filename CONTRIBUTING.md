# Contributing to Langroid

Thank you for your interest in contributing to Langroid!
We want to fundamentally change how LLM applications are built, 
using Langroid's principled multi-agent framework. 
Together, let us build the future of LLM-apps!
We welcome contributions from everyone.

Below you will find guidelines and suggestions for contributing.
We explicitly designed Langroid with a transparent, flexible architecture to 
make it easier to build LLM-powered applications, as well as 
to make it easier to contribute to Langroid itself.

# How can I Contribute?

There are many ways to contribute to Langroid. Here are some areas where you can help:

- Bug Reports
- Code Fixes
- Feature Requests
- Feature Implementations
- Documentation
- Testing
- UI/UX Improvements
- Translations
- Outreach

You are welcome to take on un-assigned open [issues](https://github.com/langroid/langroid/issues).

## Implementation Ideas

**INTEGRATIONS**

- Vector databases, e.g.: Pinecone, Milvus, Weaviate, PostgresML, ...
- Other LLM APIs, e.g.: Claude, Cohere, ...
- Local LLMs, e.g.: llama2


**SPECIALIZED AGENTS**

- `SQLChatAgent`, analogous to `DocChatAgent`: adds ability to chat with SQL databases
- `TableChatAgent`: adds ability to chat with a tabular dataset in a file. 
   This can derive from `RetrieverAgent`

**CORE LANGROID**

- Implement a way to **backtrack** 1 step in a multi-agent task. 
For instance during a long multi-agent conversation, if we receive a bad response from the LLM,
when the user gets a chance to respond, they may insert a special code (e.g. `b`) so that 
the previous step is re-done and the LLM gets another chance to respond.
- Implement Agents that communicate via REST APIs: Currently, all agents within 
the multi-agent system are created in a single script. 
We can remove this limitation, and add the ability to have agents running and 
listening to an end-point (e.g. a flask server). For example the LLM may 
generate a function-call or Langroid-tool-message, which the agent’s 
tool-handling method interprets and makes a corresponding request to an API endpoint. 
This request can be handled by an agent listening to requests at this endpoint, 
and the tool-handling method gets the result and returns it as the result of the handling method. 
This is roughly the mechanism behind OpenAI plugins, e.g. https://github.com/openai/chatgpt-retrieval-plugin

**DEMOS, POC, Use-cases**

- Data Analyst Demo: A multi-agent system that automates a data analysis workflow, e.g. 
feature-exploration, visualization, ML model training.
- Document classification based on rules in an unstructured “policy” document. 
    This is an actual use-case from a large US bank. The task is to classify 
    documents into categories “Public” or “Sensitive”. The classification must be 
    informed by a “policy” document which has various criteria. 
    Normally, someone would have to read the policy doc, and apply that to 
    classify the documents, and maybe go back and forth and look up the policy repeatedly. 
    This would be a perfect use-case for Langroid’s multi-agent system. 
    One agent would read the policy, perhaps extract the info into some structured form. 
    Another agent would apply the various criteria from the policy to the document in question, 
    and (possibly with other helper agents) classify the document, along with a detailed justification.

- Document classification and tagging: Given a collection of already labeled/tagged docs, 
which have been ingested into a vecdb (to allow semantic search), 
when given a new document to label/tag, we retrieve the most similar docs 
from multiple categories/tags from the vecdb and present these (with the labels/tags) 
as few-shot examples to the LLM, and have the LLM classify/tag the retrieved document.

- Implement the CAMEL multi-agent debate system : https://lablab.ai/t/camel-tutorial-building-communicative-agents-for-large-scale-language-model-exploration

- Implement Stanford’s Simulacra paper with Langroid.
Generative Agents: Interactive Simulacra of Human Behavior https://arxiv.org/abs/2304.03442

- Implement CMU's paper with Langroid.
Emergent autonomous scientific research capabilities of large language models https://arxiv.org/pdf/2304.05332.pdf

---

# Contribution Guidelines

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
- Cache Configs
  - Redis : Password, Host, Port <br>
    OR
  - Momento : Auth_token
- Qdrant API key for the vector database.

```bash
cp .env-template .env
# now edit the .env file, insert your secrets as above
``` 

Currently only OpenAI models are supported. 
You are welcome to submit a PR to submit other API-based or local models. 

## Run tests
To verify your env is correctly setup, run all tests using `make tests`.

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
