# Langroid: Quick-start Guide

## Install

```bash
pip install langroid
```

## Set up tokens/keys 

Langroid uses a few APIs, and you need to set up tokens/keys for these APIs.
At the very least you need an OpenAI API key. 
The need for other keys is indicated below.

- **OpenAI API** key (required): If you don't have one, see [this OpenAI Page]
  (https://help.openai.com/en/collections/3675940-getting-started-with-openai-api).
- **Qdrant** Vector Store API Key (required for apps that need retrieval from 
  documents): Sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io)
  This capacity will suffice for quite a while (even after 3 months of 
  development we've only used 30% of the capacity). Qdrant is the default 
  vector-store in Langroid. Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported. We use the local-storage version of Chroma, so there is no need for an 
  API key. 
- **GitHub** Personal Access Token (required for apps that need to analyze git 
  repos; token-based API calls are less rate-limited). See this
    [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- **Redis** Password (optional, only needed to cache LLM API responses):
  Redis [offers](https://redis.com/try-free/) a free 30MB Redis account 
  which is more than sufficient to try out Langroid and even beyond.

In the root of your repo, create a file called `.env` and add the keys in this format: 
```bash
OPENAI_API_KEY=<your key>
GITHUB_ACCESS_TOKEN=<your token>
REDIS_PASSWORD=<your password>
QDRANT_API_KEY=<your key>
```

Now you should be ready to use Langroid!
In the sections below we show you how to use the various components of 
`langroid`.


## Using the OpenAI Language Models directly
At the most basic level, you might want to directly interact with the OpenAI LLM. 



## Examples of using Langroid
There are two places to find examples of using Langroid, in the GitHub repo 
itself:

- The [tests](https://github.com/langroid/langroid/tree/main/tests/main) 
  folder under `tests/main`
- The [examples](https://github.com/langroid/langroid/tree/main/examples) 
  folder under `examples`


