---
title: 'Using Langroid with Local LLMs'
draft: true
date: 2023-09-03
authors: 
  - pchalasani
categories:
  - langroid
  - llm
  - local-llm
comments: true
---

# Using Langroid with Local LLMs

## Why local models?
There are commercial, remotely served models that currently appear to beat all open/local
models. So why care about local models? Local models are exciting for a number of reasons:

<!-- more -->

- **cost**: other than compute/electricity, there is no cost to use them.
- **privacy**: no concerns about sending your data to a remote server.
- **latency**: no network latency due to remote API calls, so faster response times, provided you can get fast enough inference.
- **uncensored**: some like the fact many local models are not censored to avoid sensitive topics.
- **fine-tunable**: you can fine-tune them on private/recent data, which current commercial models don't have access to.
- **sheer thrill**: having a model running on your machine with no internet connection, 
  and being able to have an intelligent conversation with it -- there is something almost magical about it.

The main appeal with local models is that with sufficiently careful prompting,
they may behave sufficiently well to be useful for specific tasks/domains, 
and bring all of the above benefits.

## Running LLMs locally

There are several ways to use LLMs locally. See the [`r/LocalLLaMA`](https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/) subreddit for 
a wealth of information. There are open source libraries that offer front-ends
to run local models, for example [`oobabooga/text-generation-webui`](https://github.com/oobabooga/text-generation-webui)
(or "ooba-TGW" for short) but the focus in this tutorial is on spinning up a
server that mimics an OpenAI-like API, so that any Langroid code that works with 
the OpenAI API (for say GPT3.5 or GPT4) will work with a local model,
with just a simple change: set `openai.api_base` to the URL where the local API 
server is listening, typically `http://localhost:8000/v1`.

There are two libraries we recommend for setting up local models with OpenAI-like APIs:

- [ooba-TGW](https://github.com/oobabooga/text-generation-webui) mentioned above, for a variety of models, including llama2 models.
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (LCP for short), specifically for llama2 models.


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

Once you have the above server running (e.g., in a separate terminal tab),
create another virtual env where you install langroid as usual.
Note that local models are supported from version 0.1.60 onwards. 
There are two ways to setup Langroid to use local Llama2 models:

###  Option 1: Via Environment Variables

Make sure you have set up a `.env` file as described in
the [langroid-examples](https://github.com/langroid/langroid-examples#set-up-environment-variables-api-keys-etc) repo.
Then add this variable to the `.env` file:
```bash
# modify if using non-default host, port when you set up the server above
OPENAI_LOCAL.API_BASE=http://localhost:8000/v1
```
In case you are using the non-default context length (by passing `--n_ctx` to the server),
you would need to set an additional environment variable as well, as in this example:
```bash
OPENAI_LOCAL.CONTEXT_LENGTH=1000
```
Since you are using a local model, of course the value of `OPENAI_API_KEY` is irrelevant.
You can set it to a junk value just to make sure you are not using the OpenAI API.

Now any script or test that uses Langroid will use the local model.

### Option 2: By creating config objects in Python Code

Switching to a local Llama model using environment variables is convenient because
you don't need to change any code. 
However, switching models within our Python code offers more flexibility, 
e.g., to programmatically switch 
between using a local model and the OpenAI API, for different types of tasks, or 
allow different agents to use different models. 
Of course, the two options can be combined, as noted in the comments below.

In your script where you want to use the local model,
first specify a `LocalModelConfig` object with various settings, and 
create an instance of `OpenAIGPTConfig` object from this:

```python
from langroid.language_models.base import LocalModelConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from dotenv import load_dotenv

load_dotenv()  # read in .env file to set env vars

local_model_config = LocalModelConfig(
        api_base="http://localhost:8000/v1",  # (1)! 
        context_length=1000,  # (2)!
        use_completion_for_chat=True,  # (3)
)

llm_config = OpenAIGPTConfig(local=local_model_config)
```

1. If omitted, uses the value of `OPENAI_LOCAL.API_BASE` env var
2. If omitted, uses the value of `OPENAI_LOCAL.CONTEXT_LENGTH` env var
3. If omitted, uses the value of `OPENAI_LOCAL.USE_COMPLETION_FOR_CHAT` env var. See the next section for more.

For more on the `use_completion_for_chat` flag, see the [`Chat Completion`](../blog/posts/chat-completion.md) tutorial. 

Then use this config to define a `ChatAgentConfig`, create an agent, wrap it in a Task, and run it:

```python
from langroid.agent.chat_agent import ChatAgentConfig, ChatAgent
from langroid.agent.task import Task

config = ChatAgentConfig(
        system_message="You are a helpful assistant.",
        llm=local_llm_config,
)
agent = ChatAgent(config=config)
task = Task(agent=agent)
user_message = "Hello"
task.run(user_message)
```

See a full working example of a simple command-line chatbot that you can use with either
the OpenAI GPT4 model or a local llama model, in the `langroid-examples` repo:
[https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat.py](https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat.py).

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



