# Handling a non-tool LLM message

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

Consider what happens when you invoke `task.run()`. When the agent's `llm_response` 
returns a valid tool-call, the sequence of steps looks like this:

- `llm_response` -> tool $T$
- `aggent_response` handles $T$ -> returns results $R$
- `llm_response` responds to $R$ -> returns msg $M$
- and so on

If the LLM's response M contains a valid tool, then this cycle continues
with another tool-handling round. However, if the LLM's response M does _not_ contain
a tool-call, it is unclear whether:

- (1) the LLM "forgot" to generate a tool (or generated it wrongly, hence it was
   not recognized by Langroid as a tool), or 
- (2) the LLM's response M is an "answer" meant to be shown to the user 
    to continue the conversation, or
- (3) the LLM's response M is intended to be a "final" response, ending the task. 

Internally, when the `ChatAgent`'s `agent_response` method sees a message that does not
contain a tool, it invokes the `handle_message_fallback` method, which by default
does nothing (returns `None`). However you can override this method by deriving
from `ChatAgent`, as described in this [FAQ](https://langroid.github.io/langroid/FAQ/#how-can-i-handle-an-llm-forgetting-to-generate-a-toolmessage). As in that FAQ, 
in this fallback method, you would
typically have code that checks whether the message is a `ChatDocument`
and whether it came from the LLM, and if so, you would have the method return 
an appropriate message or tool (e.g. a reminder to the LLM, or an orchestration tool
such as [`AgentDoneTool`][langroid.agent.tools.orchestration.AgentDoneTool]).

To simplify the developer experience, as of version 0.39.2 Langroid also provides an
easier way to specify what this fallback method should return, via the
`ChatAgentConfig.handle_llm_no_tool` parameter, for example:
```python
config = lr.ChatAgentConfig(
    # ... other params
    handle_llm_no_tool="done", # terminate task if LLM sends non-tool msg
)
```
The `handle_llm_no_tool` parameter can have the following possible values:

- A special value from the [`NonToolAction`][langroid.mytypes.NonToolAction] Enum, e.g.:
    - `"user"` or `NonToolAction.USER` - this is interpreted by langroid to return 
     `ForwardTool(agent="user")`, meaning the message is passed to the user to await
     their next input.
    - `"done"` or `NonToolAction.DONE` - this is interpreted by langroid to return 
     `AgentDoneTool(content=msg.content, tools=msg.tool_messages)`, 
     meaning the task is ended, and any content and tools in the current message will
     appear in the returned `ChatDocument`.
- A callable, specifically a function that takes a `ChatDocument` and returns any value. 
  This can be useful when you want the fallback action to return a value 
  based on the current message, e.g. 
  `lambda msg: AgentDoneTool(content=msg.content)`, or it could a more 
  elaborate function, or a prompt that contains the content of the current message.
- Any `ToolMessage` (typically an [Orchestration](https://github.com/langroid/langroid/blob/main/langroid/agent/tools/orchestration.py) tool like 
  `AgentDoneTool` or `ResultTool`)
- Any string, meant to be handled by the LLM. 
  Typically this would be a reminder to the LLM, something like:
```python
"""Your intent is not clear -- 
- if you forgot to use a Tool such as `ask_tool` or `search_tool`, try again.
- or if you intended to return your final answer, use the Tool named `done_tool`,
  with `content` set to your answer.
"""
```

A simple example is in the [`chat-search.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search.py) 
script, and in the `test_handle_llm_no_tool` test in   
[`test_tool_messages.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_tool_messages.py).

