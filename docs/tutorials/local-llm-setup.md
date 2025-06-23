# Setting up a Local/Open LLM to work with Langroid

!!! tip "Examples scripts in [`examples/`](https://github.com/langroid/langroid/tree/main/examples) directory."
      There are numerous examples of scripts that can be run with local LLMs,
      in the [`examples/`](https://github.com/langroid/langroid/tree/main/examples)
      directory of the main `langroid` repo. These examples are also in the 
      [`langroid-examples`](https://github.com/langroid/langroid-examples/tree/main/examples),
      although the latter repo may contain some examples that are not in the `langroid` repo.
      Most of these example scripts allow you to specify an LLM in the format `-m <model>`,
      where the specification of `<model>` is described in the quide below for local/open LLMs, 
      or in the [Non-OpenAI LLM](https://langroid.github.io/langroid/tutorials/non-openai-llms/) guide. Scripts 
      that have the string `local` in their name have been especially designed to work with 
      certain local LLMs, as described in the respective scripts.
      If you want a pointer to a specific script that illustrates a 2-agent chat, have a look 
      at [`chat-search-assistant.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-assistant.py).
      This specific script, originally designed for GPT-4/GPT-4o, works well with `llama3-70b` 
      (tested via Groq, mentioned below).

## Easiest: with Ollama

As of version 0.1.24, Ollama provides an OpenAI-compatible API server for the LLMs it supports,
which massively simplifies running these LLMs with Langroid. Example below.

```
ollama pull mistral:7b-instruct-v0.2-q8_0
```
This provides an OpenAI-compatible 
server for the `mistral:7b-instruct-v0.2-q8_0` model.

You can run any Langroid script using this model, by setting the `chat_model`
in the `OpenAIGPTConfig` to `ollama/mistral:7b-instruct-v0.2-q8_0`, e.g.

```python
import langroid.language_models as lm
import langroid as lr

llm_config = lm.OpenAIGPTConfig(
    chat_model="ollama/mistral:7b-instruct-v0.2-q8_0",
    chat_context_length=16_000, # adjust based on model
)
agent_config = lr.ChatAgentConfig(
    llm=llm_config,
    system_message="You are helpful but concise",
)
agent = lr.ChatAgent(agent_config)
# directly invoke agent's llm_response method
# response = agent.llm_response("What is the capital of Russia?")
task = lr.Task(agent, interactive=True)
task.run() # for an interactive chat loop
```

## Setup Ollama with a GGUF model from HuggingFace

Some models are not directly supported by Ollama out of the box. To server a GGUF
model with Ollama, you can download the model from HuggingFace and set up a custom
Modelfile for it.

E.g. download the GGUF version of `dolphin-mixtral` from
[here](https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF)

(specifically, download this file `dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf`)

To set up a custom ollama model based on this:

- Save this model at a convenient place, e.g. `~/.ollama/models/`
- Create a modelfile for this model. First see what an existing modelfile
  for a similar model looks like, e.g. by running:

```
ollama show --modelfile dolphin-mixtral:latest
```
You will notice this file has a FROM line followed by a prompt template and other settings.
Create a new file with these contents.
Only  change the  `FROM ...` line with the path to the model you downloaded, e.g.
```
FROM /Users/blah/.ollama/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf
```

- Save this modelfile somewhere, e.g. `~/.ollama/modelfiles/dolphin-mixtral-gguf`
- Create a new ollama model based on this file:
```
ollama create dolphin-mixtral-gguf -f ~/.ollama/modelfiles/dolphin-mixtral-gguf
``` 

- Run this new model using `ollama run dolphin-mixtral-gguf`

To use this model with Langroid you can then specify `ollama/dolphin-mixtral-gguf`
as the `chat_model` param in the `OpenAIGPTConfig` as in the previous section.
When a script supports it, you can also pass in the model name via
`-m ollama/dolphin-mixtral-gguf`

## Local LLMs using LMStudio

LMStudio is one of the simplest ways to download run open-weight LLMs locally.
See their docs at [lmstudio.ai](https://lmstudio.ai/docs) for installation and usage 
instructions. Once you download a model, you can use the "server" option to have it 
served via an OpenAI-compatible API at a local IP like `https://127.0.0.1:1234/v1`.
As with any other scenario of running a local LLM, you can use this with Langroid by
setting `chat_model` as follows (note you should not include the `https://` part):

```python
llm_config = lm.OpenAIGPTConfig(
    chat_model="local/127.0.0.1234/v1",
    ...
)
```

## Setup llama.cpp with a GGUF model from HuggingFace

See `llama.cpp`'s [GitHub page](https://github.com/ggerganov/llama.cpp/tree/master) for build and installation instructions.

After installation, begin as above with downloading a GGUF model from HuggingFace; for example, the quantized `Qwen2.5-Coder-7B` from [here](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF); specifically, [this file](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/qwen2.5-coder-7b-instruct-q2_k.gguf).

Now, the server can be started with `llama-server -m qwen2.5-coder-7b-instruct-q2_k.gguf`.

In addition, your `llama.cpp` may be built with support for simplified management of HuggingFace models (specifically, `libcurl` support is required); in this case, `llama.cpp` will download HuggingFace models to a cache directory, and the server may be run with:
```bash
llama-server \
      --hf-repo Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
      --hf-file qwen2.5-coder-7b-instruct-q2_k.gguf
```

To use the model with Langroid, specify `llamacpp/localhost:{port}` as the `chat_model`; the default port is 8080.

## Setup vLLM with a model from HuggingFace

See [the vLLM docs](https://docs.vllm.ai/en/stable/getting_started/installation.html) for installation and configuration options. To run a HuggingFace model with vLLM, use `vllm serve`, which provides an OpenAI-compatible server. 

For example, to run `Qwen2.5-Coder-32B`, run `vllm serve Qwen/Qwen2.5-Coder-32B`.

If the model is not publicly available, set the environment varaible `HF_TOKEN` to your HuggingFace token with read access to the model repo.

To use the model with Langroid, specify `vllm/Qwen/Qwen2.5-Coder-32B` as the `chat_model` and, if a port other than the default 8000 was used, set `api_base` to `localhost:{port}`.

## Setup vLLM with a GGUF model from HuggingFace

`vLLM` supports running quantized models from GGUF files; however, this is currently an experimental feature. To run a quantized `Qwen2.5-Coder-32B`, download the model from [the repo](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF), specifically [this file](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/blob/main/qwen2.5-coder-32b-instruct-q4_0.gguf). 

The model can now be run with `vllm serve qwen2.5-coder-32b-instruct-q4_0.gguf --tokenizer Qwen/Qwen2.5-Coder-32B` (the tokenizer of the base model rather than the quantized model should be used).

To use the model with Langroid, specify `vllm/qwen2.5-coder-32b-instruct-q4_0.gguf` as the `chat_model` and, if a port other than the default 8000 was used, set `api_base` to `localhost:{port}`.

## "Local" LLMs hosted on Groq
In this scenario, an open-source LLM (e.g. `llama3.1-8b-instant`) is hosted on a Groq server
which provides an OpenAI-compatible API. Using this with langroid is exactly analogous
to the Ollama scenario above: you can set the `chat_model` in the `OpenAIGPTConfig` to
`groq/<model_name>`, e.g. `groq/llama3.1-8b-instant`. 
For this to work, ensure you have a `GROQ_API_KEY` environment variable set in your
`.env` file. See [groq docs](https://console.groq.com/docs/quickstart).

## "Local" LLMs hosted on Cerebras
This works exactly like with Groq, except you set up a `CEREBRAS_API_KEY` environment variable, and specify the `chat_model` as `cerebras/<model_name>`, e.g. `cerebras/llama3.1-8b`. See the Cerebras [docs](https://inference-docs.cerebras.ai/introduction) for details on which LLMs are supported.

## Open/Proprietary LLMs via OpenRouter

OpenRouter is a **paid service** that provides an OpenAI-compatible API 
for practically any LLM, open or proprietary.
Using this with Langroid is similar to the `groq` scenario above:

- Ensure you have an `OPENROUTER_API_KEY` set up in your environment (or `.env` file), and 
- Set the `chat_model` in the `OpenAIGPTConfig` to 
  `openrouter/<model_name>`, where `<model_name>` is the name of the model on the 
[OpenRouter](https://openrouter.ai/) website, e.g. `qwen/qwen-2.5-7b-instruct`.

This is a good option if you want to use larger open LLMs without having to download
them locally (especially if your local machine does not have the resources to run them).
Besides using specific LLMs, OpenRouter also has smart routing/load-balancing.
OpenRouter is also convenient for using proprietary LLMs (e.g. gemini, amazon) via 
a single convenient API.

## Custom LLMs via AI/ML API

AI/ML API is a **flexible OpenAI-compatible platform** that supports over 300 models (e.g. DeepSeek, GPT-4o, Gemini, Claude, and more) with **enterprise-grade rate limits** and **global low-latency hosting**. See the [official model list](https://aimlapi.com/models?utm_source=langroid&utm_medium=github&utm_campaign=integration).

Using it with Langroid is simple and similar to Groq or OpenRouter:

* Set the `AIML_API_KEY` in your environment or `.env` file
* Set `chat_model` in your `OpenAIGPTConfig` to `aimlapi/<model_name>`, for example:
  `aimlapi/deepseek-chat`, `aimlapi/gpt-4o`, or `aimlapi/claude-3-sonnet`

Optionally, set a custom `api_base`, if not using the default `https://api.aimlapi.com/v1`.

This setup gives you access to powerful open and proprietary LLMs via a unified interface, ideal for:

* avoiding per-provider account setup,
* getting fast inference without local model deployment,
* switching between models via a single flag.


### 🔗 Website: [https://aimlapi.com/](https://aimlapi.com/app/?utm_source=langroid&utm_medium=github&utm_campaign=integration)

### 🔗 Docs: [https://docs.aimlapi.com/](https://docs.aimlapi.com/?utm_source=langroid&utm_medium=github&utm_campaign=integration)

---

Если нужно, добавлю вариант с CLI запуском `run_api.py` как в твоём примере.

## "Local" LLMs hosted on GLHF.chat

See [glhf.chat](https://glhf.chat/chat/create) for a list of available models.

To run with one of these models, set the `chat_model` in the `OpenAIGPTConfig` to
`"glhf/<model_name>"`, where `model_name` is `hf:` followed by the HuggingFace repo 
path, e.g. `Qwen/Qwen2.5-Coder-32B-Instruct`, so the full `chat_model` would be
`"glhf/hf:Qwen/Qwen2.5-Coder-32B-Instruct"`. 

## DeepSeek LLMs

As of 26-Dec-2024, DeepSeek models are available via their [api](https://platform.deepseek.com).
To use it with Langroid:

- set up your `DEEPSEEK_API_KEY` environment variable in the `.env` file or as
 an explicit export in your shell
- set the `chat_model` in the `OpenAIGPTConfig` to `deepseek/deepseek-chat` to use the 
`DeepSeek-V3` model, or `deepseek/deepseek-reasoner` to use the full (i.e. non-distilled) `DeepSeek-R1` "reasoning" model.

The DeepSeek models are also available via OpenRouter (see the corresponding 
in the OpenRouter section here) or ollama (see those instructions). E.g. you
can use the DeepSeek R1 or its distilled variants by setting `chat_model` to 
`openrouter/deepseek/deepseek-r1` or `ollama/deepseek-r1:8b`.

## Other non-OpenAI LLMs supported by LiteLLM

For other scenarios of running local/remote LLMs, it is possible that the `LiteLLM` library
supports an "OpenAI adaptor" for these models (see their [docs](https://litellm.vercel.app/docs/providers)).

Depending on the specific model, the `litellm` docs may say you need to 
specify a model in the form `<provider>/<model>`, e.g. `palm/chat-bison`. 
To use the model with Langroid, simply prepend `litellm/` to this string, e.g. `litellm/palm/chat-bison`,
when you specify the `chat_model` in the `OpenAIGPTConfig`.

To use `litellm`, ensure you have the `litellm` extra installed, 
via `pip install langroid[litellm]` or equivalent.



## Harder: with oobabooga
Like Ollama, [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) provides an OpenAI-API-compatible API server, but the setup 
is significantly more involved. See their github page for installation and model-download instructions.

Once you have finished the installation, you can spin up the server for an LLM using
something like this:

```
python server.py --api --model mistral-7b-instruct-v0.2.Q8_0.gguf --verbose --extensions openai --nowebui
```
This will show a message saying that the OpenAI-compatible API is running at `http://127.0.0.1:5000`

Then in your Langroid code you can specify the LLM config using
`chat_model="local/127.0.0.1:5000/v1` (the `v1` is the API version, which is required).
As with Ollama, you can use the `-m` arg in many of the example scripts, e.g.
```
python examples/docqa/rag-local-simple.py -m local/127.0.0.1:5000/v1
```

Recommended: to ensure accurate chat formatting (and not use the defaults from ooba),
  append the appropriate HuggingFace model name to the
  -m arg, separated by //, e.g. 
```
python examples/docqa/rag-local-simple.py -m local/127.0.0.1:5000/v1//mistral-instruct-v0.2
```
  (no need to include the full model name, as long as you include enough to
   uniquely identify the model's chat formatting template)


## Other local LLM scenarios

There may be scenarios where the above `local/...` or `ollama/...` syntactic shorthand
does not work.(e.g. when using vLLM to spin up a local LLM at an OpenAI-compatible
endpoint). For these scenarios, you will have to explicitly create an instance of 
`lm.OpenAIGPTConfig` and set *both* the `chat_model` and `api_base` parameters.
For example, suppose you are able to get responses from this endpoint using something like:
```bash
curl http://192.168.0.5:5078/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Mistral-7B-Instruct-v0.2",
        "messages": [
             {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```
To use this endpoint with Langroid, you would create an `OpenAIGPTConfig` like this:
```python
import langroid.language_models as lm
llm_config = lm.OpenAIGPTConfig(
    chat_model="Mistral-7B-Instruct-v0.2",
    api_base="http://192.168.0.5:5078/v1",
)
```

## Quick testing with local LLMs
As mentioned [here](https://langroid.github.io/langroid/tutorials/non-openai-llms/#quick-testing-with-non-openai-models), 
you can run many of the [tests](https://github.com/langroid/langroid/tree/main/tests/main) in the main langroid repo against a local LLM
(which by default run against an OpenAI model), 
by specifying the model as `--m <model>`, 
where `<model>` follows the syntax described in the previous sections. Here's an example:

```bash
pytest tests/main/test_chat_agent.py --m ollama/mixtral
```
Of course, bear in mind that the tests may not pass due to weaknesses of the local LLM.
