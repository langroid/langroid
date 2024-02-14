# Setting up a local LLM to work with Langroid

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
llm_config = OpenAIGPTConfig(
    chat_model="ollama/mistral:7b-instruct-v0.2-q8_0",
    chat_context_length=16_000, # adjust based on model
)
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

To use this model with Langroid you can then specify `dolphin-mixtral-gguf`
as the `chat_model` param in the `OpenAIGPTConfig` as in the previous section.
When a script supports it, you can also pass in the model name via
`-m litellm/ollama_chat/dolphin-mixtral-gguf`

## Other non-Ollama LLMs supported by LiteLLM

For other scenarios of running local/remote LLMs, it is possible that the `LiteLLM` library
supports an "OpenAI adaptor" for these models (see their [docs](https://litellm.vercel.app/docs/providers)).

Depending on the specific model, the `litellm` docs may say you need to 
specify a model in the form `<provider>/<model>`, e.g. `palm/chat-bison`. 
To use the model with Langroid, simply prepend `litellm/` to this string, e.g. `litellm/palm/chat-bison`,
when you specify the `chat_model` in the `OpenAIGPTConfig`.

To use `litellm`, ensure you have the `litellm` extra installed, 
via `pip install langroid[litellm]` or equivalent.



## Harder: with oobabooga
Like Ollama, oobabooga provides an OpenAI-API-compatible API server, but the setup 
is significantly more involved. See 
https://github.com/oobabooga/text-generation-webui for installation and model-download instructions.

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