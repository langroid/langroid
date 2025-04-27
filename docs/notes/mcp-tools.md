from langroid.agent.tools.exa_search_tool import ExaSearchTool

# Langroid MCP Integration

Langroid provides seamless integration with Model Context Protocol (MCP) servers via 
two main approaches, both of which involve creating Langroid `ToolMessage` subclasses
corresponding to the MCP tools. This integration allows _any_ LLM
(that is good enough to do function-calling via prompts) to use any MCP server.

1. **FastMCPClient** – programmatic creation of `ToolMessage` classes.
2. **`@mcp_tool` decorator** – declarative creation and optional customization.

See the following to understand the integration better:
- example python scripts under [`examples/mcp`](https://github.com/langroid/langroid/tree/main/examples/mcp)
- [`tests/main/test_mcp_tools.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_mcp_tools.py)

---

## 1. FastMCPClient

```python
from langroid.agent.tools.mcp import FastMCPClient
import asyncio

async def main() -> None:
    # server_spec can be:
    #   • Path to a local Python script
    #   • URL to a server
    #   • fastmcp.server.FastMCP instance
    #   • fastmcp.client.transports.ClientTransport (see below) 
    server_spec = "path/to/weather_server.py"
    async with FastMCPClient(server_spec) as client:
        # List all available tools
        tools = await client.get_tools()

        # Create a ToolMessage subclass for "get_alerts"
        GetAlerts = await client.make_tool("get_alerts")

    msg = GetAlerts(state="NY")

    # Call via handle_async()
    alerts = await msg.handle_async()
    print(alerts)

asyncio.run(main())
```

### Useful Methods

- `FastMCPClient(server_spec)`  
  Create a client for the given spec.

- `await client.connect()` / `await client.close()`  
  Manually open/close the session.

- `await client.get_tools() -> list[Type[ToolMessage]]`  
  Generate all `ToolMessage` subclasses.

- `await client.make_tool(tool_name: str) -> Type[ToolMessage]`  
  Build a single `ToolMessage` class.

- `await client.find_mcp_tool(name: str) -> Tool | None`  
  Retrieve raw MCP `Tool` metadata.

- `await client.call_mcp_tool(name: str, args: dict) -> str|list[str]|None`  
  Low-level call to the MCP tool.

---

## 2. `@mcp_tool` Decorator

```python
from fastmcp.server import FastMCP
from langroid.agent.tools.mcp import mcp_tool
import langroid as lr

# Define your MCP server (pydantic v2 for schema)
server = FastMCP("MyServer")

@mcp_tool(server, "greet")
class GreetTool(lr.ToolMessage):
    """Say hello to someone."""

    async def handle_async(self) -> str:
        # Customize post-processing
        raw = await self.call_tool_async()
        return f"💬 {raw}"
```

- **Arguments**
    1. `server_spec`: path/URL/`FastMCP`/`ClientTransport`
    2. `tool_name`: name of the MCP tool

- **Behavior**
    - Generates a `ToolMessage` subclass with all input fields typed.
    - Provides `call_tool_async()` under the hood.
    - If you define your own `handle_async()`, it overrides the default.

---

## 3. Customizing `handle_async`

By overriding `handle_async`, you can format, filter, or enrich the tool output 
before it’s sent back to the LLM. If you don't override it, the default behavior is to
simply return the value of the "raw" MCP tool call `await self.call_tool_async()`.

```python
@mcp_tool(server, "calculate")
class CalcTool(ToolMessage):
    """Perform complex calculation."""

    async def handle_async(self) -> str:
        result = await self.call_tool_async()
        # Add context or emojis, etc.
        return f"🧮 Result is *{result}*"
```

---

## 4. Transport Examples

Langroid supports any transport that can be defined via [FastMCP](https://gofastmcp.com/clients/transports).
Below are examples of using `NpxStdioTransport` and `UvxStdioTransport` to connect
to MCP servers and create Langroid `ToolMessage` subclasses.

### NPX Stdio Transport

```python
from fastmcp.client.transports import NpxStdioTransport
from langroid.agent.tools.mcp import FastMCPClient, mcp_tool
from langroid import ToolMessage

# 1) Define NPX transport for an external MCP server package
npx = NpxStdioTransport(
    package="exa-mcp-server",
    env_vars={"EXA_API_KEY": "..."}
)

# 2) Programmatic creation via FastMCPClient
async with FastMCPClient(npx) as client:
    WebSearch = await client.make_tool("web_search_exa")

# 3) Declarative creation via decorator
@mcp_tool(npx, "web_search_exa")
class WebSearchTool(ToolMessage):
    """Perform web searches via EXA MCP server."""
    pass
```

### UVX Stdio Transport

```python
from fastmcp.client.transports import UvxStdioTransport
from langroid.agent.tools.mcp import FastMCPClient, mcp_tool
from langroid import ToolMessage

# 1) Define UVX transport pointing to a git MCP server
uvx = UvxStdioTransport(tool_name="mcp-server-git")

# 2) Programmatic creation via FastMCPClient
async with FastMCPClient(uvx) as client:
  GitStatus = await client.make_tool("git_status")

# 3) Declarative creation via decorator and custom handler
@mcp_tool(uvx, "git_status")
class GitStatusTool(ToolMessage):
  """Get git repository status via UVX MCP server."""
  async def handle_async(self) -> str:
    status = await self.call_tool_async()
    return "GIT STATUS: " + status
```

---

## 5. Enabling Tools in Your Agent

Once you’ve created a Langroid `ToolMessage` subclass from an MCP server, enable it on a `ChatAgent`, just like you normally would. Below is an example of using 
the [Exa MCP server](https://docs.exa.ai/examples/exa-mcp) to create a 
Langroid web search tool, enable a `ChatAgent` to use it, and then set up a `Task` to 
run the agent loop.

First we must define the appropriate `ClientTransport` for the MCP server:
```python
# define the transport
transport = NpxStdioTransport(
    package="exa-mcp-server",
    env_vars=dict(EXA_API_KEY=os.getenv("EXA_API_KEY")),
)
```

Then we use the `@mcp_tool` decorator to create a `ToolMessage` 
subclass representing the web search tool. Note that one reason to use the decorator
to define our tool is so we can specify a custom `handle_async` method that
controls what is sent to the LLM after the actual raw MCP tool-call
(the `call_tool_async` method) is made.

```python
# the second arg specifically refers to the `web_search_exa` tool available
# on the server defined by the `transport` variable.
@mcp_tool(transport, "web_search_exa")
class ExaSearchTool(lr.ToolMessage):
    async def handle_async(self):
        result: str = await self.call_tool_async()
        return f"""
        Below are the results of the web search:
        
        <WebSearchResult>
        {result}
        </WebSearchResult>
        
        Use these results to answer the user's original question.
        """

```

If we did not want to override the `handle_async` method, we could simply have
created the `ExaSearchTool` class programmatically using the `FastMCPClient` class, 
as shown in the previous section, i.e.:
```python
from langroid.agent.tools.mcp import FastMCPClient

async with FastMCPClient(transport) as client:
    ExaSearchTool = await client.make_tool("web_search_exa")
```

We can now define our main function where we create our `ChatAgent`,
attach the `ExaSearchTool` to it, define the `Task`, and run the task loop.

```python
async def main():
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                # this defaults to True, but we set it to False so we can see output
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the web-search tool
    agent.enable_message(ExaSearchTool)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async()
```

See [`exa-web-search.py`](https://github.com/langroid/langroid/blob/main/examples/mcp/exa-web-search.py) for a full 
working example of this. 