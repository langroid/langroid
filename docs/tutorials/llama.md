# Using Langroid with locally running LLama2 models

## Background

There are a couple of ways to use LLama2 models:

- Using the HuggingFace `transformers` library, e.g. if you click on "Use In Transformers" 
button on [https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) you get the following 
instructions:
```python
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="TheBloke/Llama-2-13B-chat-GGUF")

OR 

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("TheBloke/Llama-2-13B-chat-GGUF")
```

In general with this approach you will need GPUs to run the model.

- Using `llama.cpp`: run quantized versions of these models locally on CPU, e.g. on your macbook.

In these notes we explore the `llama.cpp` approach. 

!!! note "Tested on Macbook M1 Pro Max"
        This has been tested only on a Macbook Pro M1 Max with 64GB RAM. 

## Install `llama-cpp-python`
The overall plan will be to spin up a server that presents an OpenAI-like API to 
the local llama model, and then use Langroid to interact with this server.

:warning: Very important to install `llama-cpp-python` in a separate virtual env 
from the one where you install Langroid.

Follow the instructions here:
[https://github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

Essentially, you can do `poetry add llama-cpp-python` or `pip install llama-cpp-python`.

As the repo says, you may need to do:

```bash
pip install "llama-cpp-python[server]" --force-reinstall --upgrade --no-cache-dir
```


## Download a model

E.g. go here:
[https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main)

Pick one of the `gguf` model files, say `llama-2-13b-chat.Q4_K_M.gguf` and click 
the download button, save the model under your `./models/` dir.

## Directly use the model in Python

```python
from llama_cpp import Llama
llm = Llama(
    model_path="./models/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf",
    n_ctx=1000, # context window
)
# since we are using a "chat" model, we can directly ask questions
llm("Name the planets in the solar system")
```

## Set up a web-server that presents an OpenAI-like API

This is more interesting for us, since this means we can directly use it with Langroid.

Then you can run the server like this:

```bash
python3 -m llama_cpp.server --model models/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf
```

Then this presents an OpenAPI doc here:

[http://localhost:8000/docs](http://localhost:8000/docs)

As you can all the usual OpenAI end-points are available here.


## Use the local model with the OpenAI library

Awesome that this actually works: You simply use the `openai` library,
pointing it to the local server `http://localhost:8000/v1`, 
give a fake OpenAI API key, and it works!

```python
import os
import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "sx-xxx" # immaterial
OPENAI_API_KEY = "sx-xxx" # immaterial
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

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

In your `.env` file, set the following:

```bash
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=sx-xxx
```

Then you can use Langroid as usual. Try some simple tests, such as:

```python
pytest -s tests/main/test_llm.py
```

!!! note "Tests May Fail, results may be inferior, apps/examples may fail!"
        Be aware that while the above enables you to use Langroid with local llama2 models,
        the tests or examples may fail, and tools and special agents may not work well, 
        if they work at all. The reason is that a lot of the code in Langroid (and for that 
        matter in any LLM framework) relies on a certain level of competence of the underlying
        LLM model. When we say "competence" we are referring to accuracy of responses, alignment,
        and ability to follow instructions. Since there is still (as of Aug 2023) a huge gap between Llama2 models
        and GPT-4, much of the code in Langroid may not work well with Llama2 models.
        It could well be that with much more explicit prompting and many more few-shot examples,
        the behavior of the agents using llama2 models can be improved. But we leave this to the 
        user to explore. Another point to note is that it is possible that llama2 behavior
        is sufficiently "smart" for certain narrowly defined tasks, but again we leave
        this to users to explore.

        



