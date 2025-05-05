# Langroid MCP Integration

Langroid provides seamless integration with Model Context Protocol (MCP) servers via 
two methods, both of which involve creating Langroid `ToolMessage` subclasses
corresponding to the MCP tools: 

1. Programmatic creation of Langroid tools using `get_langroid_tool_async`, 
   `get_langroid_tools_async` from the tool definitions defined on an MCP server.
2. Declarative creation of Langroid tools using the **`@mcp_tool` decorator**, which allows
   customizing the tool-handling behavior beyond what is provided by the MCP server.

This integration allows _any_ LLM (that is good enough to do function-calling via prompts) to use any MCP server.
See the following to understand the integration better:

- example python scripts under [`examples/mcp`](https://github.com/langroid/langroid/tree/main/examples/mcp)
- [`tests/main/test_mcp_tools.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_mcp_tools.py)

---

## 1. Connecting to an MCP server via transport specification

Before creating Langroid tools, we first need to define and connect to an MCP server
via a [FastMCP](https://gofastmcp.com/getting-started/welcome) client. 
There are several ways to connect to a server, depending on how it is defined. 
Each of these uses a different type of [transport](https://gofastmcp.com/clients/transports).

The typical pattern to use with Langroid is as follows:

- define an MCP server transport
- create a `ToolMessage` subclass using the `@mcp_tool` decorator or 
  `get_langroid_tool_async()` function, with the transport as the first argument


Langroid's MCP integration will work with any of [transports](https://gofastmcp.com/clients/transportsl) 
supported by FastMCP.
Below we go over some common ways to define transports and extract tools from the servers.

1. **Local Python script**
2. **In-memory FastMCP server** - useful for testing and for simple in-memory servers
   that don't need to be run as a separate process.
3. **NPX stdio transport**
4. **UVX stdio transport**
5. **Generic stdio transport** – launch any CLI‐based MCP server via stdin/stdout
6. **Network SSE transport** – connect to HTTP/S MCP servers via `SSETransport`


All examples below use the async helpers to create Langroid tools (`ToolMessage` subclasses):

```python
from langroid.agent.tools.mcp import (
    get_tools_async,
    get_tool_async,
)
```

---

#### Path to a Python Script

Point at your MCP‐server entrypoint, e.g., to the `weather.py` script in the 
langroid repo (based on the [Anthropic quick-start guide](https://modelcontextprotocol.io/quickstart/server)):

```python
async def example_script_path() -> None:
    server = "tests/main/mcp/weather-server-python/weather.py"
    tools = await get_langroid_tools_async(server) # all tools available
    AlertTool = await get_langroid_tool_async(server, "get_alerts") # specific tool

    # instantiate the tool with a specific input
    msg = AlertTool(state="CA")
    
    # Call the tool via handle_async()
    alerts = await msg.handle_async()
    print(alerts)
```

---

#### In-Memory FastMCP Server

Define your server with `FastMCP(...)` and pass the instance:

```python
from fastmcp.server import FastMCP
from pydantic import BaseModel, Field

class CounterInput(BaseModel):
    start: int = Field(...)

def make_server() -> FastMCP:
    server = FastMCP("CounterServer")

    @server.tool()
    def increment(data: CounterInput) -> int:
        """Increment start by 1."""
        return data.start + 1

    return server

async def example_in_memory() -> None:
    server = make_server()
    tools = await get_langroid_tools_async(server)
    IncTool = await get_langroid_tool_async(server, "increment")

    result = await IncTool(start=41).handle_async()
    print(result)  # 42
```

See the [`mcp-file-system.py`](https://github.com/langroid/langroid/blob/main/examples/mcp/mcp-file-system.py)
script for a working example of this.

---

#### NPX stdio Transport

Use any npm-installed MCP server via `npx`, e.g., the 
[Exa web-search MCP server](https://docs.exa.ai/examples/exa-mcp):

```python
from fastmcp.client.transports import NpxStdioTransport

transport = NpxStdioTransport(
    package="exa-mcp-server",
    env_vars={"EXA_API_KEY": "…"},
)

async def example_npx() -> None:
    tools = await get_langroid_tools_async(transport)
    SearchTool = await get_langroid_tool_async(transport, "web_search_exa")

    results = await SearchTool(
        query="How does Langroid integrate with MCP?"
    ).handle_async()
    print(results)
```

For a fully working example, see the script [`exa-web-search.py`](https://github.com/langroid/langroid/blob/main/examples/mcp/exa-web-search.py).

---

#### UVX stdio Transport

Connect to a UVX-based MCP server, e.g., the [Git MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/git)

```python
from fastmcp.client.transports import UvxStdioTransport

transport = UvxStdioTransport(tool_name="mcp-server-git")

async def example_uvx() -> None:
    tools = await get_langroid_tools_async(transport)
    GitStatus = await get_langroid_tool_async(transport, "git_status")

    status = await GitStatus(path=".").handle_async()
    print(status)
```

#### Generic stdio Transport

Use `StdioTransport` to run any MCP server as a subprocess over stdio:

```python
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp import get_tools_async,

get_tool_async


async def example_stdio() -> None:
    """Example: any CLI‐based MCP server via StdioTransport."""
    transport: StdioTransport = StdioTransport(
        command="uv",
        args=["run", "--with", "biomcp-python", "biomcp", "run"],
    )
    tools: list[type] = await get_tools_async(transport)
    BioTool = await get_tool_async(transport, "tool_name")
    result: str = await BioTool(param="value").handle_async()
    print(result)
```

See the full example in [`examples/mcp/biomcp.py`](https://github.com/langroid/langroid/blob/main/examples/mcp/biomcp.py).

#### Network SSE Transport

Use `SSETransport` to connect to a FastMCP server over HTTP/S:

```python
from fastmcp.client.transports import SSETransport
from langroid.agent.tools.mcp import (
    get_tools_async,
    get_tool_async,
)


async def example_sse() -> None:
    """Example: connect to an HTTP/S MCP server via SSETransport."""
    url: str = "https://localhost:8000/sse"
    transport: SSETransport = SSETransport(
        url=url, headers={"Authorization": "Bearer TOKEN"}
    )
    tools: list[type] = await get_tools_async(transport)
    ExampleTool = await get_tool_async(transport, "tool_name")
    result: str = await ExampleTool(param="value").handle_async()
    print(result)
```    
---

With these patterns you can list tools, generate Pydantic-backed `ToolMessage` classes,
and invoke them via `.handle_async()`, all with zero boilerplate client setup.


---

## 2. Create Langroid Tools declaratively using the `@mcp_tool` decorator

The above examples showed how you can create Langroid tools programmatically using
the helper functions `get_langroid_tool_async()` and `get_langroid_tools_async()`,
with the first argument being the transport to the MCP server. The `@mcp_tool` decorator
works in the same way: 

- **Arguments to the decorator**
    1. `server_spec`: path/URL/`FastMCP`/`ClientTransport`, as mentioned above.
    2. `tool_name`: name of a specific MCP tool

- **Behavior**
    - Generates a `ToolMessage` subclass with all input fields typed.
    - Provides a `call_tool_async()` under the hood -- this is the "raw" MCP tool call,
      returning a string.
    - If you define your own `handle_async()`, it overrides the default. Typically,
you would override it to customize either the input or the output of the tool call, or both.
    - If you don't define your own `handle_async()`, it defaults to just returning the
      value of the `call_tool_async()` method.

Here is a simple example of using the `@mcp_tool` decorator to create a Langroid tool:

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

Using the decorator method allows you to customize the `handle_async` method of the
tool, or add additional fields to the `ToolMessage`. 
You may want to customize the input to the tool, or the tool result before it is sent back to 
the LLM. If you don't override it, the default behavior is to simply return the value of 
the "raw" MCP tool call `await self.call_tool_async()`. 

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

## 3. Enabling Tools in Your Agent

Once you’ve created a Langroid `ToolMessage` subclass from an MCP server, 
you can enable it on a `ChatAgent`, just like you normally would. Below is an example of using 
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
created the `ExaSearchTool` class programmatically via the `get_langroid_tool_async` 
function as shown above, i.e.:

```python
from langroid.agent.tools.mcp import get_tool_async

ExaSearchTool = awwait
get_tool_async(transport, "web_search_exa")
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