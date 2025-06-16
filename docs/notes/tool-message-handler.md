# Tool Message Handlers in Langroid

## Overview

Langroid provides flexible ways to define handlers for `ToolMessage` classes. When a tool is used by an LLM, the framework needs to know how to handle it. This can be done either by defining a handler method in the `Agent` class or within the `ToolMessage` class itself.

## Enabling Tools with `enable_message`

Before an agent can use or handle a tool, it must be explicitly enabled using the `enable_message` method. This method takes two important arguments:

- **`use`** (bool): Whether the LLM is allowed to generate this tool
- **`handle`** (bool): Whether the agent is allowed to handle this tool

```python
# Enable both generation and handling (default)
agent.enable_message(MyTool, use=True, handle=True)

# Enable only handling (agent can handle but LLM won't generate)
agent.enable_message(MyTool, use=False, handle=True)

# Enable only generation (LLM can generate but agent won't handle)
agent.enable_message(MyTool, use=True, handle=False)
```

When `handle=True` and the `ToolMessage` has a `handle` method defined, this method is inserted into the agent with a name matching the tool's `request` field value. This insertion only happens when `enable_message` is called.

## Default Handler Mechanism

By default, `ToolMessage` uses and/or creates a handler in `Agent` class instance with the name identical to the tool's `request` attribute.

### Agent-based Handlers
If a tool `MyTool` has `request` attribute `my_tool`, you can define a method `my_tool` in your `Agent` class that will handle this tool when the LLM generates it:

```python
class MyTool(ToolMessage):
    request = "my_tool"
    param: str

class MyAgent(ChatAgent):
    def my_tool(self, msg: MyTool) -> str:
        return f"Handled: {msg.param}"

# Enable the tool
agent = MyAgent()
agent.enable_message(MyTool)
```

### ToolMessage-based Handlers
Alternatively, if a tool is "stateless" (i.e. does not require the Agent's state), you can define a `handle` method within the `ToolMessage` class itself. When you call `enable_message` with `handle=True`, Langroid will insert this method into the `Agent` with the name matching the `request` field value:

```python
class MyTool(ToolMessage):
    request = "my_tool"
    param: str
    
    def handle(self) -> str:
        return f"Handled: {self.param}"

# Enable the tool
agent = MyAgent()
agent.enable_message(MyTool)  # The handle method is now inserted as "my_tool" in the agent
```

## Flexible Handler Signatures

Handler methods (`handle()` or `handle_async()`) support multiple signature patterns to access different levels of context:

### 1. No Arguments (Simple Handler)
This is the typical pattern for stateless tools that do not require any context from 
the agent or current chat document.

```python
class MyTool(ToolMessage):
    request = "my_tool"
    
    def handle(self) -> str:
        return "Simple response"
```

### 2. Agent Parameter Only
Use this pattern when you need access to the `Agent` instance, 
but not the current chat document.
```python
from langroid.agent.base import Agent

class MyTool(ToolMessage):
    request = "my_tool"
    
    def handle(self, agent: Agent) -> str:
        return f"Response from {agent.name}"
```

### 3. ChatDocument Parameter Only
Use this pattern when you need access to the current `ChatDocument`,
but not the `Agent` instance.
```python
from langroid.agent.chat_document import ChatDocument

class MyTool(ToolMessage):
    request = "my_tool"
    
    def handle(self, chat_doc: ChatDocument) -> str:
        return f"Responding to: {chat_doc.content}"
```

### 4. Both Agent and ChatDocument Parameters
This is the most flexible pattern, allowing access to both the `Agent` instance
and the current `ChatDocument`. The order of parameters does not matter, but
as noted below, it is highly recommended to always use type annotations.
```python
class MyTool(ToolMessage):
    request = "my_tool"
    
    def handle(self, agent: Agent, chat_doc: ChatDocument) -> ChatDocument:
        return agent.create_agent_response(
            content="Response with full context",
            files=[...]  # Optional file attachments
        )
```

## Parameter Detection

The framework automatically detects handler parameter types through:

1. **Type annotations** (recommended): The framework uses type hints to determine which parameters to pass
2. **Parameter names** (fallback): If no type annotations are present, it looks for parameters named `agent` or `chat_doc`

It is highly recommended to always use type annotations for clarity and reliability.

### Example with Type Annotations (Recommended)
```python
def handle(self, agent: Agent, chat_doc: ChatDocument) -> str:
    # Framework knows to pass both agent and chat_doc
    return "Handled"
```

### Example without Type Annotations (Not Recommended)
```python
def handle(self, agent, chat_doc):  # Works but not recommended
    # Framework uses parameter names to determine what to pass
    return "Handled"
```

## Async Handlers

All the above patterns also work with async handlers:

```python
class MyTool(ToolMessage):
    request = "my_tool"
    
    async def handle_async(self, agent: Agent) -> str:
        # Async operations here
        result = await some_async_operation()
        return f"Async result: {result}"
```

See the quick-start [Tool section](https://langroid.github.io/langroid/quick-start/chat-agent-tool/) for more details.

## Custom Handler Names

In some use-cases it may be beneficial to separate the 
*name of a tool* (i.e. the value of `request` attribute) from the 
*name of the handler method*. 
For example, you may be dynamically creating tools based on some data from
external data sources. Or you may want to use the same "handler" method for
multiple tools.

This may be done by adding `_handler` attribute to the `ToolMessage` class,
that defines name of the tool handler method in `Agent` class instance.
The underscore `_` prefix ensures that the `_handler` attribute does not 
appear in the Pydantic-based JSON schema of the `ToolMessage` class, 
and so the LLM would not be instructed to generate it.

!!! note "`_handler` and `handle`"
    A `ToolMessage` may have a `handle` method defined within the class itself,
    as mentioned above, and this should not be confused with the `_handler` attribute.

For example:
```
class MyToolMessage(ToolMessage):
    request: str = "my_tool"
    _handler: str = "tool_handler"

class MyAgent(ChatAgent):
    def tool_handler(
        self,
        message: ToolMessage,
    ) -> str:
        if tool.request == "my_tool":
            # do something
```

Refer to [examples/basic/tool-custom-handler.py](https://github.com/langroid/langroid/blob/main/examples/basic/tool-custom-handler.py)
for a detailed example.
