# Defining tool handler with custom name

By default `ToolMessage` uses and/or creates a handler in `Agent` class
instance with the name identical to the tool's `request` attribute.

In some use-cases it may be benefitial to change this default behavior.
For example, you may be dynamically creating tools based on some data from
external data sources. Or you may want to use the same "handler" method for
multiple tools.

This may be done by adding `_handle` attribute to the `ToolMessage` class,
that defines name of the tool handler method in `Agent` class instance.

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

Refer to examples/basic/tool-custom-handler.py for a more thorough example.
