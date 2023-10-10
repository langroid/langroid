```bash
pip install langroid
```
```python

from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT

# create the (Pydantic-derived) config class: Allows setting params via MYLLM_XXX env vars
MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm") 

# instantiate the class, with the model name and context length
my_llm_config = MyLLMConfig(
    chat_model="local/localhost:8000", # "local/[URL where LiteLLM proxy is listening]
    chat_context_length=2048,  # adjust based on model
)

# create llm and interact with it 
from langroid.language_models.base import LLMMessage, Role

llm = OpenAIGPT(my_llm_config)
messages = [
    LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM),
    LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
],
response = mdl.chat(messages, max_tokens=50)

# Create an Agent with this LLM, wrap it in a Task, and run it as an interactive chat app:
from langroid.agent.base import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task

agent_config = ChatAgentConfig(llm=my_llm_config, name="my-llm-agent")
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run() 
```