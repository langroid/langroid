---
title: 'Using Langroid with Local LLMs'
draft: false
date: 2023-09-14
authors: 
  - pchalasani
categories:
  - langroid
  - llm
  - local-llm
comments: true
---
## Why local models?
There are commercial, remotely served models that currently appear to beat all open/local
models. So why care about local models? Local models are exciting for a number of reasons:

<!-- more -->

- **cost**: other than compute/electricity, there is no cost to use them.
- **privacy**: no concerns about sending your data to a remote server.
- **latency**: no network latency due to remote API calls, so faster response times, provided you can get fast enough inference.
- **uncensored**: some local models are not censored to avoid sensitive topics.
- **fine-tunable**: you can fine-tune them on private/recent data, which current commercial models don't have access to.
- **sheer thrill**: having a model running on your machine with no internet connection,
  and being able to have an intelligent conversation with it -- there is something almost magical about it.

The main appeal with local models is that with sufficiently careful prompting,
they may behave sufficiently well to be useful for specific tasks/domains,
and bring all of the above benefits. Some ideas on how you might use local LLMs:

- In a multi-agent system, you could have some agents use local models for narrow 
  tasks with a lower bar for accuracy (and fix responses with multiple tries).
- You could run many instances of the same or different models and combine their responses.
- Local LLMs can act as a privacy layer, to identify and handle sensitive data before passing to remote LLMs.
- Some local LLMs have intriguing features, for example llama.cpp lets you 
  constrain its output using a grammar.

## Running LLMs locally

There are several ways to use LLMs locally. See the [`r/LocalLLaMA`](https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/) subreddit for
a wealth of information. There are open source libraries that offer front-ends
to run local models, for example [`oobabooga/text-generation-webui`](https://github.com/oobabooga/text-generation-webui)
(or "ooba-TGW" for short) but the focus in this tutorial is on spinning up a
server that mimics an OpenAI-like API, so that any code that works with
the OpenAI API (for say GPT3.5 or GPT4) will work with a local model,
with just a simple change: set `openai.api_base` to the URL where the local API
server is listening, typically `http://localhost:8000/v1`.

There are a few libraries we recommend for setting up local models with OpenAI-like APIs:

- [LiteLLM OpenAI Proxy Server](https://docs.litellm.ai/docs/proxy_server) lets you set up a local 
  proxy server for over 100+ LLM providers (remote and local).
- [ooba-TGW](https://github.com/oobabooga/text-generation-webui) mentioned above, for a variety of models, including llama2 models.
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (LCP for short), specifically for llama2 models.
- [ollama](https://github.com/jmorganca/ollama)

We recommend visiting these links to see how to install and run these libraries.

## Use the local model with the OpenAI library

Once you have a server running using any of the above methods, 
your code that works with the OpenAI models can be made to work 
with the local model, by simply changing the `openai.api_base` to the 
URL where the local server is listening. 

If you are using Langroid to build LLM applications, the framework takes
care of the `api_base` setting in most cases, and you need to only set
the `chat_model` parameter in the LLM config object for the LLM model you are using.
See the [Non-OpenAI LLM tutorial](../../tutorials/non-openai-llms.md) for more details.



<iframe src="https://langroid.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>



