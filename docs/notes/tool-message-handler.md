# Defining tool handler with custom name

By default `ToolMessage` uses and/or creates a handler in `Agent` class
instance with the name identical to the tool's `request` attribute.
E.g. if a tool `MyTool` has `request` attribute `my_tool`, then 
one can define a method `my_tool` in `MyAgent`, that will handle this tool
when the LLM generates it. Alternatively, if `MyTool` is "stateless"
(i.e. does not require the Agent's state), then one can define a `handle` method
within the `MyTool` class itself, and Langroid will insert this method into the `Agent`,
with name `my_tool`. 
See the quick-start [Tool section](https://langroid.github.io/langroid/quick-start/chat-agent-tool/) for more details.

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
