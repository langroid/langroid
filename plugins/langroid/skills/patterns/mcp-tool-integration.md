# MCP Tool Integration Pattern

Enable Langroid agents to use tools from MCP (Model Context Protocol) servers,
such as Claude Code's file editing tools.

## Key Imports

```python
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp import mcp_tool
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
import langroid as lr
```

## Setting Up the Transport

Connect to an MCP server via stdio (e.g., Claude Code):

```python
transport = StdioTransport(
    command="claude",
    args=["mcp", "serve"],
    env={},
)
```

## Option 1: Enable ALL Tools from MCP Server

Use `get_tools_async()` to fetch and enable all available tools:

```python
async def setup_agent_with_all_tools():
    all_tools = await get_tools_async(transport)

    agent = lr.ChatAgent(lr.ChatAgentConfig(
        system_message="You have access to file tools.",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    ))

    agent.enable_message(all_tools)  # Enable all tools at once
    return agent
```

## Option 2: Enable SPECIFIC Tools (Preferred)

Use the `@mcp_tool` decorator to create ToolMessage subclasses for specific
tools. This gives you control over which tools are available and lets you
customize result handling.

```python
# Basic usage - just wrap the MCP tool
@mcp_tool(transport, "Read")
class ReadTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()


@mcp_tool(transport, "Edit")
class EditTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()


@mcp_tool(transport, "Write")
class WriteTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()


# Enable specific tools on agent
agent.enable_message(ReadTool)
agent.enable_message(EditTool)
agent.enable_message(WriteTool)
```

## Option 3: Custom Result Processing

Override `handle_async()` to transform MCP tool results before returning to LLM:

```python
@mcp_tool(transport, "Grep")
class GrepTool(lr.ToolMessage):
    async def handle_async(self):
        result = await self.call_tool_async()

        # Result may be tuple (text, files) or just text
        result_text, _files = result if isinstance(result, tuple) else (result, [])

        # Parse and transform the result
        import json
        try:
            data = json.loads(result_text)
            # Custom formatting...
            return f"Found {data.get('numMatches', 0)} matches:\n{data.get('content', '')}"
        except:
            return result_text
```

## Complete Example: File Editor Agent

```python
from fastmcp.client.transports import StdioTransport
from langroid.agent.tools.mcp import mcp_tool
import langroid as lr

transport = StdioTransport(
    command="claude",
    args=["mcp", "serve"],
    env={},
)


@mcp_tool(transport, "Read")
class ReadFileTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()


@mcp_tool(transport, "Edit")
class EditFileTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()


async def create_file_editor_agent():
    agent = lr.ChatAgent(lr.ChatAgentConfig(
        name="FileEditor",
        system_message="""You are a file editor. Use the Read tool to read files
        and the Edit tool to make changes.""",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    ))

    agent.enable_message(ReadFileTool)
    agent.enable_message(EditFileTool)

    return agent


async def main():
    agent = await create_file_editor_agent()
    task = lr.Task(agent, interactive=False)

    result = await task.run_async(
        "Read the file proposal.md and fix any typos you find."
    )
    return result
```

## Server Factory Pattern (for Concurrency)

For concurrent usage, create fresh transports to avoid `ClosedResourceError`:

```python
def make_transport():
    return StdioTransport(
        command="claude",
        args=["mcp", "serve"],
        env={},
    )

# Use factory when creating tools for concurrent scenarios
@mcp_tool(make_transport, "Edit")  # Pass factory, not instance
class EditTool(lr.ToolMessage):
    async def handle_async(self):
        return await self.call_tool_async()
```

## Available Claude Code MCP Tools

Common tools exposed by Claude Code's MCP server:

- `Read` - Read file contents
- `Edit` - Edit file with old_string/new_string replacement
- `Write` - Write/create files
- `Grep` - Search with ripgrep
- `Glob` - Find files by pattern
- `Bash` - Execute shell commands
- `LS` - List directory contents
