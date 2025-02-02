# Routing a non-tool LLM message

A common scenario is to define a `ChatAgent`, enable it to use some tools
(i.e. `ToolMessages`s), wrap it in a Task, and call `task.run()`, e.g. 

```python
class MyTool(lr.ToolMessage)
    ...
    
import langroid as lr
config = lr.ChatAgentConfig(...)
agent = lr.ChatAgent(config)
agent.enable_message(MyTool)
task = lr.Task(agent, interactive=False)
task.run("Hello")
```

Consider the steps involved in `task.run()`. When the agent's `llm_response` 
returns a valid tool-call, the sequence of steps looks like this:

- `llm_response` -> tool $T$
- `aggent_response` handles $T$ -> returns results $R$
- `llm_response` responds to $R$ -> returns msg $M$
- and so on

If the LLM's response M contains a valid tool, then this cycle continues
with another tool-handling round. However, if the LLM's response M does _not_ contain
a tool-call, it is unclear whether:

- (1) the LLM "forgot" to generate a tool, or
- (2) the LLM's response M is an "answer" meant to be shown to the user 
    to continue the conversation, or
- (3) the LLM's response M is intended to be a "final" response, ending the task. 

To handle such `non-tool` LLM responses, we can override the `ChatAgent`'s
`handle_message_fallback` method, as described in  
this [FAQ](https://langroid.github.io/langroid/FAQ/#how-can-i-handle-an-llm-forgetting-to-generate-a-toolmessage).
But in many cases we can be pretty certain that the only possibilities are (2) or (3).
For such cases Langroid provides a simpler way to specify which of those "routing"
actions to take, instead of having to explicitly define a `handle_message_fallback` 
method. In the `ChatAgentConfig` you can specify a `non_tool_routing` attribute, which
(currently) can be either "user" or "done", e.g.,

```python
config = lr.ChatAgentConfig(
    ...
    non_tool_routing="user", # or "done", or None (default)
)
```

- If this config attribute is set to `"user"`, then whenever the LLM's response has no 
tool-call, the message is forwarded to the user, awaiting their response.
- When it is set to `"done"`, then the task is ended, with the content of the result
set to the content of the last LLM response. 

A simple example is in the [`chat-search.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search.py) 
script, and a in the `test_non_tool_routing` test in   
[`test_tool_messages.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_tool_messages.py).

Behind the scenes, Langroid uses this `non_tool_routing` attribute to define
the appropriate actions in the agent's `handle_message_fallback` method.
