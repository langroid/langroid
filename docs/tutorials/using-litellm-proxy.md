# Using the `litellm` proxy server for local models.

LiteLLM offers a [proxy server](https://docs.litellm.ai/docs/proxy_server) 
that can be used to spin up a local server running an LLM, behind an OpenAI-compatible API. 
This means your code that works with OpenAI's `openai` python client will continue to work, 
by simply changing the `openai.api_base` to point to the local server. 
Here is how you can use Langroid to work with this proxy.

First in a separate terminal window, spin up a local model using `litellm`, 
e.g., `litellm --model ollama/llama2`.
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
MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm")

# instantiate the class, with the model name and context length
my_llm_config = MyLLMConfig(
    chat_model="local/localhost:8000",
    chat_context_length=2048,  # adjust based on model
)
```

You can then create an `OpenAIGPT` instance using this config object
and interact with it as usual.
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

agent_config = ChatAgentConfig(
    llm=my_llm_config,
    name="my-llm-agent",
    vecdb=None, # or a suitable vector db config
)
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run()
```




