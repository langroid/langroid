Title: Fix @mcp_tool pattern for fastmcp>=2.13 / mcp>=1.21

Date: 2025-11-07

Summary

The `@mcp_tool` decorator in Langroid currently accepts a concrete
`ClientTransport` (e.g., `StdioTransport`) created at module import time and
uses it to (a) open a short-lived connection to read the tool schema and (b)
later open a new connection when the tool is actually invoked. This pattern
works with older fastmcp/mcp, but with fastmcp≥2.13.0.2 and mcp≥1.21.0 the
transport instance becomes single-use after the first connection closes,
leading to `anyio.ClosedResourceError` when we try to reuse it.

Key files reviewed

- examples/mcp/claude-code-mcp-single.py
- langroid/agent/tools/mcp/decorators.py
- langroid/agent/tools/mcp/fastmcp_client.py

What happens at decorator time vs tool invocation time

Decorator time (module import):

- The decorator `@mcp_tool(server, tool_name)` runs immediately when the module
  is imported.
- `decorators.py` calls `get_tool(server, tool_name)` (sync wrapper) which
  `asyncio.run`s `get_tool_async`.
- `fastmcp_client.get_tool_async` does `async with FastMCPClient(server)`, which
  constructs an inner `fastmcp.client.Client(server)` and opens a session to the
  MCP server to fetch the tool definition (schema, description, etc.).
- A dynamic `ToolMessage` subclass is created with fields from the tool’s
  input schema. The class is annotated with `_client_config` that includes the
  original `server` argument so it can open a connection again later when the
  tool is invoked.
- The temporary client context is exited, closing the underlying session and
  transport.

Tool invocation time (at runtime in the agent):

- The tool’s `handle_async` calls the generated `call_tool_async`.
- `call_tool_async` reconstructs a new `FastMCPClient(**_client_config)` and
  opens a fresh connection to call `session.call_tool(...)`.

Why ClosedResourceError appears with newer fastmcp/mcp

- In our examples we pass a concrete `ClientTransport` instance to the
  decorator, e.g., a module-level `StdioTransport(...)`.
- At decorator time, we make one connection using that instance and then close
  it when exiting the client context.
- Later at tool invocation time, the generated tool tries to reuse the very same
  `ClientTransport` instance to open a second connection. With
  fastmcp≥2.13.0.2/mcp≥1.21.0 the transport object is effectively single-use
  and owns AnyIO channels/process handles that are closed when the first client
  context exits. Reusing it causes the session’s write side to be closed during
  `session.initialize()`, which surfaces as `anyio.ClosedResourceError` while
  sending the initial JSON-RPC request.
- Older versions (fastmcp==2.3.4, mcp==1.9.0) tolerated reusing the same
  transport instance, as the transport behaved more like a stateless “spec” or
  was internally recreated per connection. That leniency is gone in the newer
  stack, where transports manage lifecycle-bound resources tied to a single
  session.

Conclusion: passing a live, already-used `ClientTransport` instance through the
decorator leads to reusing a closed transport when the tool is actually
invoked, which triggers `ClosedResourceError` during session initialization.

Recommended fixes (choose one)

1) Pass a transport factory (or a server spec), not an instance

Create a zero-arg callable that returns a fresh transport each time. This keeps
the decorator pattern but ensures a brand-new transport is used for every
connection.

Example change to example file:

```python
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp import mcp_tool

def transport_factory():
    return StdioTransport(command="claude", args=["mcp", "serve"], env={})

@mcp_tool(transport_factory, "Grep")
class GrepTool(lr.ToolMessage):
    async def handle_async(self):
        result = await self.call_tool_async()
        return f"<GrepResult>\n{result}\n</GrepResult>"
```

This works because each call path (`get_tool_async` at decorator time and
`call_tool_async` at runtime) gets a fresh transport by calling the factory.

2) Defer tool creation to runtime (avoid decorator entirely)

For scripts already running inside an event loop or when you want to avoid all
import-time side effects, use the async helper instead of the decorator:

```python
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp.fastmcp_client import get_tool_async

async def main():
    BaseGrepTool = await get_tool_async(
        lambda: StdioTransport(command="claude", args=["mcp", "serve"], env={}),
        "Grep",
    )

    class GrepTool(BaseGrepTool):
        async def handle_async(self):
            result = await self.call_tool_async()
            return f"<GrepResult>\n{result}\n</GrepResult>"
```

3) Library-level hardening in Langroid (recommended)

Make Langroid resilient regardless of how callers pass `server` by allowing a
factory and by cloning transports when a live instance is provided.

Proposed changes (illustrative, not yet applied):

In `langroid/agent/tools/mcp/fastmcp_client.py`:

```python
from typing import Callable, Union
import inspect
from fastmcp.client.transports import ClientTransport

# Accept either a spec or a zero-arg factory returning a spec
ServerSpec = Union[str, FastMCP[Any], AnyUrl, ClientTransport, Callable[[], Union[str, FastMCP[Any], AnyUrl, ClientTransport]]]

class FastMCPClient:
    def __init__(self, server: ServerSpec, ...):
        self.server = server

    async def __aenter__(self) -> "FastMCPClient":
        server_spec = self.server() if callable(self.server) else self.server
        self._cm = Client(server_spec, ...)
        self.client = await self._cm.__aenter__()
        return self

    async def get_tool_async(self, tool_name: str) -> Type[ToolMessage]:
        ...
        def _as_factory(srv: ServerSpec):
            if callable(srv):
                return srv
            if isinstance(srv, ClientTransport):
                cls = srv.__class__
                sig = inspect.signature(cls)
                # build kwargs from attribute names that match ctor params
                kwargs = {
                    n: getattr(srv, n)
                    for n, p in sig.parameters.items()
                    if n != "self" and hasattr(srv, n)
                }
                return lambda: cls(**kwargs)
            return lambda: srv  # strings/URLs/FastMCP pass-through

        client_config = {
            "server": _as_factory(self.server),  # always a factory now
            ...
        }

        async def call_tool_async(itself: ToolMessage) -> Any:
            cfg = getattr(itself.__class__, "_client_config")
            server_factory = cfg["server"]
            async with FastMCPClient(server_factory, ...) as client:
                return await client.call_mcp_tool(itself.request, payload)
```

With this change:

- Callers may pass a transport instance, a factory, a URL, or a string. We
  always store a factory on the generated class, ensuring a fresh transport for
  each connection.
- `__aenter__` transparently supports receiving a factory and calling it.

Why this addresses the error

- The failure arises from reusing a closed `ClientTransport`. By switching to a
  factory-or-spec approach, every connection uses a brand-new transport
  instance, so the AnyIO channels and subprocess handles are valid during
  `session.initialize()` and the handshake completes normally.

Notes on behavior changes between versions

- The newer fastmcp/mcp stack ties the transport’s resources to the client
  context more strictly (e.g., AnyIO memory channels/process lifetime tied to
  the session). Reusing a transport object after the session is closed now fails
  early in `initialize()` with a closed writer, surfacing as
  `anyio.ClosedResourceError`.
- Older versions were more permissive about reusing the same instance, which is
  why the import-time decorator usage “accidentally” worked.

Action items

- Update examples to pass a factory to `@mcp_tool` (Option 1), or switch those
  examples to `get_tool_async` at runtime (Option 2).
- Optionally harden Langroid per Option 3 so user code keeps working even when
  a transport instance is passed.

Appendix: example patch to the failing example

```diff
--- a/examples/mcp/claude-code-mcp-single.py
+++ b/examples/mcp/claude-code-mcp-single.py
@@
-transport = StdioTransport(
-    command="claude",
-    args=["mcp", "serve"],
-    env={},
-)
+def transport_factory():
+    return StdioTransport(
+        command="claude",
+        args=["mcp", "serve"],
+        env={},
+    )

@@
-@mcp_tool(transport, "Grep")
+@mcp_tool(transport_factory, "Grep")
 class GrepTool(lr.ToolMessage):
     async def handle_async(self):
         # call the actual tool
         result: str = await self.call_tool_async()
```

