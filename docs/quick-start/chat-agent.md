# A simple chat agent

!!! tip "Script in `langroid-examples`"
        A full working example for the material in this section is
        in the `chat-agent.py` script in the `langroid-examples` repo:
        [`examples/quick-start/chat-agent.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/chat-agent.py).

## Agents and Tasks

A [`ChatAgent`](/reference/langroid/agent/chat_agent) is an abstraction that 
wraps a few components, including:
- an LLM (`ChatAgent.llm`), possibly equipped with tools/function-calling,
- optionally a vector-database (`ChatAgent.vecdb`)
The `ChatAgent` class also maintains the LLM conversation history.




A `ChatAgent` class has various methods that can be called t