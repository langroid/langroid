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
[`langroid-examples`](https://github.com/langroid/langroid-exmaples) repo, which can be a good starting point for your own repo.
The `langroid-examples` repo already contains a `pyproject.toml` file so that you can 
use `Poetry` to manage your virtual environment and dependencies. 
For example you can do 
```bash
poetry install # installs latest version of langroid
```
Alternatively, use `pip` to install `langroid` into your virtual environment:
```bash
pip install langroid
```

??? note "Optional Installs for using SQL Chat with a PostgreSQL DB"
    If you are using `SQLChatAgent`
    (e.g. the script [`examples/data-qa/sql-chat/sql_chat.py`](examples/data-qa/sql-chat/sql_chat.py)),
    with a postgres db, you will need to:
    
    - Install PostgreSQL dev libraries for your platform, e.g.
        - `sudo apt-get install libpq-dev` on Ubuntu,
        - `brew install postgresql` on Mac, etc.
    - Install langroid with the postgres extra, e.g. `pip install langroid[postgres]`
      or `poetry add langroid[postgres]` or `poetry install -E postgres`.
      If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.


!!! tip "Work in a nice terminal, such as Iterm2, rather than a notebook"
    All of the examples we will go through are command-line applications.
    For the best experience we recommend you work in a nice terminal that supports 
    colored outputs, such as [Iterm2](https://iterm2.com/).    

!!! note "OpenAI GPT4 is required"
    The various LLM prompts and instructions in Langroid 
    have been tested to work well with GPT4.
    Switching to GPT3.5-Turbo is easy via a config flag, and may suffice 
    for some applications, but in general you may see inferior results.

## Set up tokens/keys 

To get started, all you need is an OpenAI API Key.
If you don't have one, see [this OpenAI Page](https://help.openai.com/en/collections/3675940-getting-started-with-openai-api).
Currently only OpenAI models are supported. Others will be added later
(Pull Requests welcome!).

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
  You can sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io).
  If you skip setting up these, Langroid will use Qdrant in local-storage mode.
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported.
  We use the local-storage version of Chroma, so there is no need for an API key.
  Langroid uses Qdrant by default.
- **Redis** Password, host, port: This is optional, and only needed to cache LLM API responses
  using Redis Cloud. Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Langroid and even beyond.
  If you don't set up these, Langroid will use a pure-python
  Redis in-memory cache via the [Fakeredis](https://fakeredis.readthedocs.io/en/latest/) library.
- **Momento** Serverless Caching of LLM API responses (as an alternative to Redis).
  To use Momento instead of Redis:
    - enter your Momento Token in the `.env` file, as the value of `MOMENTO_AUTH_TOKEN` (see example file below),
    - in the `.env` file set `CACHE_TYPE=momento` (instead of `CACHE_TYPE=redis` which is the default).
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
  [`tests/main/test_google_search_tool.py`](tests/main/test_google_search_tool.py) to see how to use it.


If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
CACHE_TYPE=redis # or momento
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
MOMENTO_AUTH_TOKEN=your-momento-token-no-quotes # instead of REDIS* variables
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-cse-id
```

### Microsoft Azure OpenAI setup
In the root of the repo, copy the `.azure_env_template` file to a new file `.azure_env`: 

```bash
cp .azure_env_template .azure_env
```

The file `.azure_env` contains four environment variables that are required to use Azure OpenAI: `AZURE_API_KEY`, `OPENAI_API_BASE`, `OPENAI_API_VERSION`, and `OPENAI_DEPLOYMENT_NAME`.

Follow these steps from [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#retrieve-key-and-endpoint) to find these variables.

Then insert your Azure API Key. 
Your `.azure_env` file should look like this:
```bash
AZURE_API_KEY=your-key-here-without-quotes
````

## Next steps

Now you should be ready to use Langroid!
As a next step, you may want to see how you can use Langroid to [interact 
directly with the LLM](llm-interaction.md) (OpenAI GPT models only for now).









