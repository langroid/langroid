# Pattern: MCP Tool Integration

## Problem

You want your Langroid agent to use tools from an MCP (Model Context Protocol)
server, such as Claude Code's file editing tools.

## Solution

Use the `@mcp_tool` decorator for specific tools, or `get_tools_async()`
to enable all tools from an MCP server.

## Complete Code Example

```python
import asyncio
import langroid as lr
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp import mcp_tool
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async


# --- Transport Setup ---

def make_transport():
    """Factory function for fresh transport instances."""
    return StdioTransport(
        command="claude",
        args=["mcp", "serve"],
        env={},
    )


# --- Option 1: Specific Tools with @mcp_tool ---

@mcp_tool(make_transport, "Read")
class ReadFileTool(lr.ToolMessage):
    """MCP tool to read file contents."""

    async def handle_async(self):
        """Execute the MCP tool and return result."""
        result = await self.call_tool_async()
        # Result may be (text, files) tuple or just text
        if isinstance(result, tuple):
            return result[0]
        return result


@mcp_tool(make_transport, "Edit")
class EditFileTool(lr.ToolMessage):
    """MCP tool to edit files."""

    async def handle_async(self):
        result = await self.call_tool_async()
        if isinstance(result, tuple):
            return result[0]
        return result


async def run_with_specific_tools():
    """Use specific MCP tools."""
    agent = lr.ChatAgent(lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="You can read and edit files.",
    ))

    agent.enable_message(ReadFileTool)
    agent.enable_message(EditFileTool)

    task = lr.Task(agent, interactive=False)
    await task.run_async("Read the file config.py and summarize it.")


# --- Option 2: All Tools with get_tools_async ---

async def run_with_all_tools():
    """Enable all tools from MCP server."""
    transport = make_transport()
    all_tools = await get_tools_async(transport)

    agent = lr.ChatAgent(lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="You have access to file tools.",
        handle_llm_no_tool="You must use one of your tools!",
    ))

    agent.enable_message(all_tools)

    task = lr.Task(agent, interactive=False)
    await task.run_async("List files in the current directory.")


# --- Option 3: Custom Result Processing ---

@mcp_tool(make_transport, "Grep")
class GrepTool(lr.ToolMessage):
    """MCP tool with custom result processing."""

    async def handle_async(self):
        result = await self.call_tool_async()
        text = result[0] if isinstance(result, tuple) else result

        # Custom processing
        import json
        try:
            data = json.loads(text)
            matches = data.get("numMatches", 0)
            return f"Found {matches} matches:\n{data.get('content', '')}"
        except:
            return text


# --- Main ---

if __name__ == "__main__":
    asyncio.run(run_with_specific_tools())
```

## Transport Factory Pattern

For concurrent usage, use a factory function to create fresh transports:

```python
# Good: Factory function
def make_transport():
    return StdioTransport(command="claude", args=["mcp", "serve"])

@mcp_tool(make_transport, "Read")  # Pass factory, not instance
class ReadTool(lr.ToolMessage):
    ...


# Bad: Single instance (can cause ClosedResourceError)
transport = StdioTransport(command="claude", args=["mcp", "serve"])

@mcp_tool(transport, "Read")  # Don't do this for concurrent use
class ReadTool(lr.ToolMessage):
    ...
```

## Fire CLI Integration

When using Fire CLI with async main:

```python
async def main(model: str = "gpt-4o"):
    # async code here
    pass


if __name__ == "__main__":
    from fire import Fire

    def run_main(**kwargs):
        asyncio.run(main(**kwargs))

    Fire(run_main)  # Wrap async main
```

## Key Points

- Use `@mcp_tool(transport, "ToolName")` for specific tools
- Use `get_tools_async(transport)` for all tools
- Override `handle_async()` for custom result processing
- Use transport factory for concurrent usage
- Handle tuple results: `(text, files)` or just `text`
- MCP tools are async - use `run_async()` for tasks

## When to Use

- Integrating Claude Code's file tools (Read, Edit, Write, Grep)
- Connecting to any MCP-compatible server
- Adding external tool capabilities to Langroid agents
- Customizing how tool results are processed
