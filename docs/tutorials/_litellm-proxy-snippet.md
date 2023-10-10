```bash
pip install langroid
```
```python

from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT
from langroid.language_models.base import LLMMessage, Role


llm_config = OpenAIGPTConfig(
    # "format: local/[URL where LiteLLM proxy is listening]
    chat_model="local/localhost:8000", 
    chat_context_length=2048,  # adjust based on model
)

llm = OpenAIGPT(llm_config)
messages = [
    LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM),
    LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
]

# direct interaction with LLM
response = llm.chat(messages, max_tokens=50)

# Create an Agent, wrap it in a Task, run an interactive chat app:
from langroid.agent.base import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task

agent_config = ChatAgentConfig(llm=llm_config, name="my-llm-agent")
agent = ChatAgent(agent_config)

task = Task(agent, name="my-llm-task")
task.run() 
```