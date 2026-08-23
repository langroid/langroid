# Tool `_handler` Resolution

A `ToolMessage` subclass may declare `_handler` to route itself to an agent
method whose name differs from the tool's `request` value:

```python
class RedirectTool(ToolMessage):
    request: str = "redirect_tool"
    purpose: str = "Routes to a custom handler name"
    _handler = "my_custom_handler"


class MyAgent(ChatAgent):
    def my_custom_handler(self, msg: RedirectTool) -> str:
        ...
```

`_handler` is a **class-level declaration by the developer**, and is now
resolved from the tool's class rather than from the parsed instance.

## Why this matters (issue #1106)

`ToolMessage` sets `extra="allow"`, so keys the tool does not declare are
kept on the parsed instance and are readable via `getattr`. Dispatch
previously read `_handler` off the instance, so tool JSON emitted by the
LLM could carry its own `_handler` value:

```json
{"request": "safe_tool", "x": 1, "_handler": "some_other_agent_method"}
```

That parsed as `SafeTool`, but dispatch then invoked
`some_other_agent_method` instead of the tool's own handler — letting a
prompt-injected or otherwise adversarial LLM reach an agent method that
was never exposed as a tool. Resolving `_handler` from the class closes
this: LLM-supplied instance keys are ignored, since a tool call can only
select among the handlers the developer wired up.

This mirrors how the other framework-internal markers are resolved —
`_tainted` (see [tool-origin-taint](tool-origin-taint.md)) and the
tool-policy exemption marker are likewise read from the class, never from
an instance, for the same reason.

## Pydantic private attributes

Pydantic v2 represents a class-level underscore attribute as a
`ModelPrivateAttr` wrapper on the class, so
`langroid.agent.tool_message.handler_name()` unwraps it to obtain the
declared string. A side effect of the previous class-level lookup in
`enable_message` not doing this: a tool declaring **both** a `handle()`
method and a custom `_handler` raised

```
TypeError: attribute name must be string, not 'ModelPrivateAttr'
```

That combination now works.

## What is unchanged

- Tools without `_handler` are handled by the method named after their
  `request` value, as before.
- Developer-declared `_handler` redirects work exactly as before, on both
  the sync and async dispatch paths.
