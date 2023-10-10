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

- In a mult-agent system, you could have some agents use local models for narrow 
  tasks with a lower bar for accuracy (and fix responses with multiple tries).
- You could run many instances of the same or different models and combine their responses.
- Local LLMs can act as a privacy layer, to identify and handle sensitive data before passing to remote LLMs.
- Some local LLMs have intriguing features, for example llama.cpp lets you 
  constrain its output using grammars.

## Running LLMs locally

There are several ways to use LLMs locally. See the [`r/LocalLLaMA`](https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/) subreddit for
a wealth of information. There are open source libraries that offer front-ends
to run local models, for example [`oobabooga/text-generation-webui`](https://github.com/oobabooga/text-generation-webui)
(or "ooba-TGW" for short) but the focus in this tutorial is on spinning up a
server that mimics an OpenAI-like API, so that any Langroid code that works with
the OpenAI API (for say GPT3.5 or GPT4) will work with a local model,
with just a simple change: set `openai.api_base` to the URL where the local API
server is listening, typically `http://localhost:8000/v1`.

There are a few libraries we recommend for setting up local models with OpenAI-like APIs:

- [ooba-TGW](https://github.com/oobabooga/text-generation-webui) mentioned above, for a variety of models, including llama2 models.
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (LCP for short), specifically for llama2 models.
- [ollama](https://github.com/jmorganca/ollama)


Here we show instructions for `llama-cpp-python`. The process for `text-generation-webui` is similar.
Although the instructions specifically mention `llama2` models,
the same process should work for other local models as well (for example using ooba-TGW)
as long as you are able to spin up a server that mimics the OpenAI API.
As mentioned above, all you need to do is set `openai.api_base` to the URL where the local API
server is listening.

## Set up a local llama2 model server using `llama-cpp-python`

!!! warning "Keep the server, client virtual envs separate"
    Very important to install `llama-cpp-python` in a separate virtual env
    from the one where you install Langroid.

Install `llama-cpp-python` as described in this repo:
[https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
Mainly, you just need to do this (various optional settings are mentioned in
the repo, but you can ignore those for a basic example).

```bash
pip install "llama-cpp-python[server]" --force-reinstall --upgrade --no-cache-dir
```

Next, download a model from the HuggingFace model hub, for example:
[https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main)

Pick one of the `gguf` model files, say `llama-2-13b-chat.Q4_K_M.gguf` and click
the download button, save the model under your `./models/` dir.
To be able to use the model in "chat" mode, you will need one of the models
with the word `chat` or `instruct` in the name.

Now you can setup a web-server that presents an OpenAI-like API to this model:

```bash
python3 -m llama_cpp.server --model models/llama-2-13b-chat.Q4_K_M.gguf 
```
There are various command-line options you can give here, see the full list
by running:
```bash
python3 -m llama_cpp.server --help
```
We highlight some of the options here:
```bash
--n_ctx N_CTX         The context size. (default: 2048)
--host HOST           Listen address (default: localhost)
--port PORT           Listen port (default: 8000)
```
Then this presents an OpenAPI doc here:

[http://localhost:8000/docs](http://localhost:8000/docs)

As you can see, all the usual OpenAI end-points are available here.

## Use the local model with the OpenAI library

Awesome that this actually works: You simply use the `openai` library,
pointing it to the local server `http://localhost:8000/v1`,
give a fake OpenAI API key, and it works!

```python
import os
import openai
# modify host, port if you changed the defaults when running the server
openai.api_base = "http://localhost:8000/v1"

completion = openai.ChatCompletion.create(
  model="mydemo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Belgium?"}
  ]
)

print(completion.choices[0].message.content)
```

## Use the locally running Llama2 with Langroid

Please see the script
[`examples/basic/chat.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat.py)
for an example of how to setup langroid to use a local llama2 model.


!!! warning "Tests May Fail, results may be inferior, apps/examples may fail!"
    Be aware that while the above enables you to use Langroid with local llama2 models,
    the tests or examples may fail, and tools and special agents may not work well,
    if they work at all. The reason is that a lot of the code in Langroid (and for that
    matter in any LLM framework) relies on a certain level of competence of the underlying
    LLM model. When we say "competence" we are referring to accuracy of responses, alignment,
    and ability to follow instructions. Since there is still (as of Aug 2023) a huge gap between Llama2 models
    and GPT-4, much of the code in Langroid may not work well with Llama2 models.
    It could well be that with much more explicit prompting and many more few-shot examples,
    the behavior of the agents using llama2 models can be improved, especially on specific tasks or domains.
    But we leave this to the user to explore.



<iframe src="https://langroid.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>



