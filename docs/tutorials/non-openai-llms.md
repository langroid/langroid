# Using Langroid with Non-OpenAI LLMs

Langroid was initially written to work with OpenAI models via their API.
This may sound limiting, but fortunately there are tools that provide an OpenAI-like API 
for _hundreds_ of LLM providers.  We show below how you can define an `OpenAIGPTConfig` object
for these scenarios. This config object then be used to create a Langroid 
LLM object to interact with directly, or you can wrap it into an Agent and a Task
to create a chat loop or a more complex multi-agent setup where different agents may be using
different LLMs.

## LiteLLM OpenAI Proxy Server
LiteLLM is an excellent library which, among other things
(see below), offers a [proxy server](https://docs.litellm.ai/docs/proxy_server) that allows you 
spin up a server acting as a proxy for a variety of LLM models (over a 100 providers!) at an 
OpenAI-like endpoint. This means you can continue to use the `openai` python client, 
except that you will need to change the `openai.api_base` to point to the proxy server's URL
(this is done behind the scenes in Langroid via the `chat_model` name as shown below).
Here are the specifics steps to use this proxy server with Langroid:

First in a separate terminal window, spin up the proxy server `litellm`.
For example to use the `anthropic.claude-instant-v1` model, you can do:
```bash
export ANTHROPIC_API_KEY=my-api-key
litellm --model claude-instant-1
```
Or if you want to use the proxy server for a local model running with [`ollama`](https://github.com/jmorganca/ollama),
you can first run `ollama pull mistral` for example and then 
run `litellm --model ollama/mistral` to spin up the proxy server for this model.
```bash
This will show a message indicating the URL where the server is listening, e.g.,
```bash
Uvicorn running on http://0.0.0.0:8000
```

This URL is equivalent to `http://localhost:8000`, which is the URL
you will use in your Langroid code below.
To use this model in your Langroid code, first create config object for
this model and instantiate it.

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT

# create the (Pydantic-derived) config class: Allows setting params via MYLLM_XXX env vars
MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm") #(1)!

# instantiate the class, with the model name and context length
my_llm_config = MyLLMConfig(
    chat_model="local/localhost:8000",
    chat_context_length=2048,  # adjust based on model
)
```

1. The prefix ensures you can specify the settings in the instantiated object
   using environment variables (or in the `.env` file), using the `MYLLM_` prefix.
   This helps when you want to have different agents use
   different models, each with their own environment settings. If you create
   subclasses of `OpenAIGPTConfig` with different prefixes, you can set the
   environment variables for each of these models separately, and have all of these
   in the `.env` file without any conflicts.

## Other local LLM servers
There are other ways to spin up a local server running an LLM behind an OpenAI-compatible API,

- [`oobabooga/text-generation-webui`](https://github.com/oobabooga/text-generation-webui/tree/main/extensions), `ollama`, and `llama-cpp-python`.
- [`ollama`](https://github.com/jmorganca/ollama)
- [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)

For all of these, the process is the same as in the above example, i.e., you will
set the `chat_model` to a string that looks like `local/localhost:<port>` or 
`local/localhost:<port>/v1` (depending on the model). 

## Using the LiteLLM library

LiteLLM also has a [python library](https://docs.litellm.ai/docs/providers) that 
provides functions that mimic the OpenAI API
for a variety of LLMs. This means that instead of using `openai.ChatCompletion.create`,
you can use liteLLM's corresponding `completion` function, and the rest of your code
can remain the same (of course this is handled behind the scenes in Langroid, as you see below).
Also, there is no need to spin up a local server,
which is useful in some scenarios, especially when you want to have multiple
agents using different LLMs. Using the LiteLLM library with Langroid is very simple: 
simply set the `chat_agent` in the `OpenAIGPTConfig` to a string like 
`litellm/bedrock/anthropic.claude-instant-v1`:

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig
LiteLLMBedrockConfig = OpenAIGPTConfig.create(prefix="BEDROCK") 
litellm_bedrock_config = LiteLLMBedrockConfig(
    chat_model="litellm/bedrock/anthropic.claude-instant-v1", #(1)!
    chat_context_length=4096, # adjust according to model
)
```

1. This three-part model name denotes that we are using the `litellm` adapter library, 
    the LLM provider is `bedrock` and the actual model is `anthropic.claude-instant-v1`.


The general rule for the `chat_model` parameter is to prepend `litellm/` to the model name
specified in the [LiteLLM docs](https://docs.litellm.ai/docs/providers). 
For non-local models you will also need to specify one or more API Keys and related values. 
There is an internal validation function that will check if the keys for the model
have been specified in the environment variables. If not, it will raise an exception telling 
you which keys to specify. 

If you are using environment variables or a `.env` file, you can specify these 
variables using the upper-case version of the `prefix` argument to the `OpenAIGPTConfig.create`,
e.g. in the above case, you would set the following environment variables like
`BEDROCK_API_KEY=<your-api-key>`.

The `LiteLLM` library can also be used when you have a **locally-served model,**
but you are not using the `LiteLLM` proxy server. In this case you would set the 
`chat_model` parameter in the `OpenAIGPTConfig` to a string like `litellm/ollama/mistral`,
again following the pattern of prepending `litellm/` to the model name specified in the
[LiteLLM docs](https://docs.litellm.ai/docs/providers).

## Working with the created `OpenAIGPTConfig` object

Once you create an `OpenAIGPTConfig` object using any of the above methods, 
you can use it to create an object of class `OpenAIGPT` (which represents any
LLM with an OpenAI-compatible API) and interact with it directly:
```python
from langroid.language_models.base import LLMMessage, Role

llm = OpenAIGPT(my_llm_config)
messages = [
    LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM),
    LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
],
response = mdl.chat(messages, max_tokens=50)
```

When you interact directly with the LLM, you are responsible for keeping dialog history.
Also you would often want an LLM to have access to tools/functions and external
data/documents (e.g. vector DB or traditional DB). An Agent class simplifies managing all of these.
For example, you can create an Agent powered by the above LLM, wrap it in a Task and have it
run as an interactive chat app:

```python
from langroid.agent.base import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task

agent_config = ChatAgentConfig(llm=my_llm_config, name="my-llm-agent")
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run()
```

## Working example: Simple Chat script with a local/remote model

For a working example, see the [basic chat script](https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat.py)
in the `langroid-examples` repo, 
which you can run a few different ways, to interact with a non-OpenAI model in an interactive chat loop.
(If you omit the `-m` option, it will use the default OpenAI GPT-4 model.) 

1. Using the `liteLLM` proxy server, with `ollama`:
First run [`ollama`](https://github.com/jmorganca/ollama) to download and serve a local model, say `mistral`: 
```bash
ollama run mistral # download and run the mistral model
```
Then in a separate terminal window, run the liteLLM proxy server:
```bash
litellm --model ollama/mistral # run the proxy server, listening at localhost:8000
```
In a third terminal window, run the chat script:
```bash
python3 examples/basic/chat.py -m local/localhost:8000
```

2. Using the `liteLLM` library, with a remote model:
```bash
python3 examples/basic/chat.py -m litellm/bedrock/anthropic.claude-instant-v1
```

## Quick testing with non-OpenAI models

There are numerous tests in the main [Langroid repo](https://github.com/langroid/langroid) that involve
LLMs, and once you setup the dev environment as described in the README of the repo, 
you can run any of those tests (which run against the default GPT4 model) against
local/remote models that are proxied by `liteLLM` (or served locally via the options mentioned above,
such as `oobabooga`, `ollama` or `llama-cpp-python`), using the `--m <model-name>` option,
where `model-name` takes one of the forms above. Some examples of tests are:

```bash
pytest -s tests/test_llm.py --m local/localhost:8000
pytest -s tests/test_llm.py --m litellm/bedrock/anthropic.claude-instant-v1
pytest -s tests/test_llm.py --m litellm/ollama/mistral
```
When the `--m` option is omitted, the default OpenAI GPT4 model is used.

!!! note "`chat_context_length` is not affected by `--m`"
      Be aware that the `--m` only switches the model, but does not affect the `chat_context_length` 
      parameter in the `OpenAIGPTConfig` object. which you may need to adjust for different models.
      So this option is only meant for quickly testing against different models, and not meant as
      a way to switch between models in a production environment.








    