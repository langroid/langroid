# Setting up a local LLM to work with Langroid

## Easiest: with Ollama

```
ollama pull mistral:7b-instruct-v0.2-q8_0
```
This provides an API server for the LLM. However, this API is _not_ OpenAI-compatible,
so Langroid's code (which is written to "talk" to any API that is OpenAI-API-compatible)
will not work directly with the Ollama API. 
Fortunately, we can use the `litellm` library for this. 
Ensure you have the `litellm` extra installed, via `pip install langroid[litellm]` or equivalent.

Now in any Langroid script you can specify your LLM config as
```
OpenAIGPTConfig(
    chat_model="litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0",
    chat_context_length=8000, # adjust based on model
)
```
For convenience, many of the example scripts have a `-m` arg that accepts a model name,
e.g. 
```
python3 examples/basic/chat-local.py -m litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0
```

## Setup Ollama with a GGUF model from HuggingFace

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


## Harder: with oobabooga
Unlike Ollama, oobabooga provides an OpenAI-API-compatible API server, see 
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