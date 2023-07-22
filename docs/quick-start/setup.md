# Setup


## Install
Ensure you are using Python 3.11. It is best to work in a virtual environment:

```bash
# go to your repo root (which may be langroid-examples)
cd <your repo root>
python3 -m venv .venv
. ./.venv/bin/activate
```
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

Langroid uses a few APIs, and you need to set up tokens/keys for these APIs.
At the very least you need an OpenAI API key. 
The need for other keys is indicated below.

In the root of the repo, copy the `.env-template` file to a new file `.env`:
```bash
cp .env-template .env
```
Then insert your OpenAI API Key. If you don't have one, see [this OpenAI Page](https://help.openai.com/en/collections/3675940-getting-started-with-openai-api).
Your `.env` file should look like this:

```bash
OPENAI_API_KEY=your-key-here-without-quotes
````

Currently only OpenAI models are supported. Others will be added later
(Pull Requests welcome!).

All of the below are optional and not strictly needed to run any of the examples.

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  You can sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io).
  If you skip setting up these, Langroid will use Qdrant in local-storage mode.
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

If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
```

## Next steps

Now you should be ready to use Langroid!
As a next step, you may want to see how you can use Langroid to [interact 
directly with the LLM](llm-interaction.md) (OpenAI GPT models only for now).









