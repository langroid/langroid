# Setup


## Install
Ensure you are using Python 3.11. It is best to work in a virtual environment:

```bash
# go to your repo root (which may be langroid-examples)
cd <your repo root>
python3 -m venv .venv
. ./.venv/bin/activate
```
To see how to use Langroid in your own repo, you can take a look at the
[`langroid-examples`](https://github.com/langroid/langroid-examples) repo, which can be a good starting point for your own repo, 
or use the [`langroid-template`](https://github.com/langroid/langroid-template) repo.
These repos contain a `pyproject.toml` file suitable for use with the [`uv`](https://docs.astral.sh/uv/) dependency manager. After installing `uv` you can 
set up your virtual env, activate it, and install langroid into your venv like this:

```bash
uv venv --python 3.11
. ./.venv/bin/activate 
uv sync
```

Alternatively, use `pip` to install `langroid` into your virtual environment:
```bash
pip install langroid
```

The core Langroid package lets you use OpenAI Embeddings models via their API.
If you instead want to use the `sentence-transformers` embedding models from HuggingFace,
install Langroid like this:
```bash
pip install "langroid[hf-embeddings]"
```
For many practical scenarios, you may need additional optional dependencies:
- To use various document-parsers, install langroid with the `doc-chat` extra:
    ```bash
    pip install "langroid[doc-chat]"
    ```
- For "chat with databases", use the `db` extra:
    ```bash
    pip install "langroid[db]"
    ``
- You can specify multiple extras by separating them with commas, e.g.:
    ```bash
    pip install "langroid[doc-chat,db]"
    ```
- To simply install _all_ optional dependencies, use the `all` extra (but note that this will result in longer load/startup times and a larger install size):
    ```bash
    pip install "langroid[all]"
    ```

??? note "Optional Installs for using SQL Chat with a PostgreSQL DB"
    If you are using `SQLChatAgent`
    (e.g. the script [`examples/data-qa/sql-chat/sql_chat.py`](https://github.com/langroid/langroid/blob/main/examples/data-qa/sql-chat/sql_chat.py),
    with a postgres db, you will need to:
    
    - Install PostgreSQL dev libraries for your platform, e.g.
        - `sudo apt-get install libpq-dev` on Ubuntu,
        - `brew install postgresql` on Mac, etc.
    - Install langroid with the postgres extra, e.g. `pip install langroid[postgres]`
      or `uv add "langroid[postgres]"` or `uv pip install --extra postgres -r pyproject.toml`.
      If this gives you an error, try 
      `uv pip install psycopg2-binary` in your virtualenv.


!!! tip "Work in a nice terminal, such as Iterm2, rather than a notebook"
    All of the examples we will go through are command-line applications.
    For the best experience we recommend you work in a nice terminal that supports 
    colored outputs, such as [Iterm2](https://iterm2.com/).    


!!! note "mysqlclient errors"
    If you get strange errors involving `mysqlclient`, try doing `pip uninstall mysqlclient` followed by `pip install mysqlclient` 

## Set up tokens/keys 

To get started, all you need is an OpenAI API Key.
If you don't have one, see [this OpenAI Page](https://platform.openai.com/docs/quickstart).
(Note that while this is the simplest way to get started, Langroid works with practically any LLM, not just those from OpenAI.
See the guides to using [Open/Local LLMs](https://langroid.github.io/langroid/tutorials/local-llm-setup/),
and other [non-OpenAI](https://langroid.github.io/langroid/tutorials/non-openai-llms/) proprietary LLMs.)

In the root of the repo, copy the `.env-template` file to a new file `.env`:
```bash
cp .env-template .env
```
Then insert your OpenAI API Key.
Your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
```

Alternatively, you can set this as an environment variable in your shell
(you will need to do this every time you open a new shell):
```bash
export OPENAI_API_KEY=your-key-here-without-quotes
```

All of the following environment variable settings are optional, and some are only needed
to use specific features (as noted below).

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  Langroid uses LanceDB as the default vector store in its `DocChatAgent` class (for RAG).
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported.
  We use the local-storage version of Chroma, so there is no need for an API key.
- **Redis** Password, host, port: This is optional, and only needed to cache LLM API responses
  using Redis Cloud. Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Langroid and even beyond.
  If you don't set up these, Langroid will use a pure-python
  Redis in-memory cache via the [Fakeredis](https://fakeredis.readthedocs.io/en/latest/) library.
- **GitHub** Personal Access Token (required for apps that need to analyze git
  repos; token-based API calls are less rate-limited). See this
  [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- **Google Custom Search API Credentials:** Only needed to enable an Agent to use the `GoogleSearchTool`.
  To use Google Search as an LLM Tool/Plugin/function-call,
  you'll need to set up
  [a Google API key](https://developers.google.com/custom-search/v1/introduction#identify_your_application_to_google_with_api_key),
  then [setup a Google Custom Search Engine (CSE) and get the CSE ID](https://developers.google.com/custom-search/docs/tutorial/creatingcse).
  (Documentation for these can be challenging, we suggest asking GPT4 for a step-by-step guide.)
  After obtaining these credentials, store them as values of
  `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in your `.env` file.
  Full documentation on using this (and other such "stateless" tools) is coming soon, but
  in the meantime take a peek at the test
  [`tests/main/test_web_search_tools.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_web_search_tools.py) to see how to use it.


If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
CACHE_TYPE=redis
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-cse-id
```

### Microsoft Azure OpenAI setup[Optional]

This section applies only if you are using Microsoft Azure OpenAI.

When using Azure OpenAI, additional environment variables are required in the
`.env` file.
This page [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#environment-variables)
provides more information, and you can set each environment variable as follows:

- `AZURE_OPENAI_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your_resource.openai.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is an OPTIONAL deployment name which may be 
   defined by the user during the model setup.
- `AZURE_OPENAI_CHAT_MODEL` Azure OpenAI allows specific model names when you select the model for your deployment. You need to put precisely the exact model name that was selected. For example, GPT-3.5 (should be `gpt-35-turbo-16k` or `gpt-35-turbo`) or GPT-4 (should be `gpt-4-32k` or `gpt-4`).
- `AZURE_OPENAI_MODEL_NAME` (Deprecated, use `AZURE_OPENAI_CHAT_MODEL` instead).
  
!!! note "For Azure-based models use `AzureConfig` instead of `OpenAIGPTConfig`"
    In most of the docs you will see that LLMs are configured using `OpenAIGPTConfig`.
    However if you want to use Azure-deployed models, you should replace `OpenAIGPTConfig` with `AzureConfig`. See 
    the [`test_azure_openai.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_azure_openai.py) and 
    [`example/basic/chat.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat.py)


## Next steps

Now you should be ready to use Langroid!
As a next step, you may want to see how you can use Langroid to [interact 
directly with the LLM](llm-interaction.md) (OpenAI GPT models only for now).









