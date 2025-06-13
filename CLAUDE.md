# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- Install core dependencies: `pip install -e .`
- Install dev dependencies: `pip install -e ".[dev]"`
- Install specific feature groups:
  - Document chat features: `pip install -e ".[doc-chat]"`
  - Database features: `pip install -e ".[db]"`
  - HuggingFace embeddings: `pip install -e ".[hf-embeddings]"`
  - All features: `pip install -e ".[all]"`
- Run linting and type checking: `make check`
- Format code: `make lint`

### Testing
- Run all tests: `pytest tests/`
- Run specific test: `pytest tests/main/test_file.py::test_function`
- Run tests with coverage: `pytest --cov=langroid tests/`
- Run only main tests: `make tests` (uses `pytest tests/main`)

### Linting and Type Checking
- Lint code: `make check` (runs black, ruff check, mypy)
- Format only: `make lint` (runs black and ruff fix)
- Type check only: `make type-check`
- Always use `make check` to run lints + mypy before trying to commit changes

### Version and Release Management
- Bump version: `./bump_version.sh [patch|minor|major]`
- Or use make commands:
  - `make all-patch` - Bump patch version, build, push, release
  - `make all-minor` - Bump minor version, build, push, release
  - `make all-major` - Bump major version, build, push, release

## Architecture

Langroid is a framework for building LLM-powered agents that can use tools and collaborate with each other.

### Core Components:

1. **Agents** (`langroid/agent/`):
   - `chat_agent.py` - Base ChatAgent that can converse and use tools
   - `task.py` - Handles execution flow for agents
   - `special/` - Domain-specific agents (doc chat, table chat, SQL chat, etc.)
   - `openai_assistant.py` - Integration with OpenAI Assistant API

2. **Tools** (`langroid/agent/tools/`):
   - Tool system for agents to interact with external systems
   - `tool_message.py` - Protocol for tool messages
   - Various search tools (Google, DuckDuckGo, Tavily, Exa, etc.)

3. **Language Models** (`langroid/language_models/`):
   - Abstract interfaces for different LLM providers
   - Implementations for OpenAI, Azure, local models, etc.
   - Support for hundreds of LLMs via LiteLLM

4. **Vector Stores** (`langroid/vector_store/`):
   - Abstract interface and implementations for different vector databases
   - Includes support for Qdrant, Chroma, LanceDB, Pinecone, PGVector, Weaviate

5. **Document Processing** (`langroid/parsing/`):
   - Parse and process documents from various formats
   - Chunk text for embedding and retrieval
   - Support for PDF, DOCX, images, and more

6. **Embedding Models** (`langroid/embedding_models/`):
   - Abstract interface for embedding generation
   - Support for OpenAI, HuggingFace, and custom embeddings

### Key Multi-Agent Patterns:

- **Task Delegation**: Agents can delegate tasks to other agents through hierarchical task structures
- **Message Passing**: Agents communicate by transforming and passing messages
- **Collaboration**: Multiple agents can work together on complex tasks

### Key Security Features:

- The `full_eval` flag in both `TableChatAgentConfig` and `VectorStoreConfig` controls code injection protection
- Defaults to `False` for security, set to `True` only in trusted environments

## Documentation

- Main documentation is in the `docs/` directory
- Examples in the `examples/` directory demonstrate usage patterns
- Quick start examples available in `examples/quick-start/`

## MCP (Model Context Protocol) Tools Integration

Langroid provides comprehensive support for MCP tools through the `langroid.agent.tools.mcp` module. Here are the key patterns and approaches:

### MCP Tool Creation Methods

#### 1. Using the `@mcp_tool` Decorator (Module Level)
```python
from langroid.agent.tools.mcp import mcp_tool
from fastmcp.client.transports import StdioTransport

transport = StdioTransport(command="...", args=[...])

@mcp_tool(transport, "tool_name")
class MyTool(lr.ToolMessage):
    async def handle_async(self):
        result = await self.call_tool_async()
        # custom processing
        return result
```

**Important**: The decorator creates the transport connection at module import time, so it must be used at module level (not inside async functions).

#### 2. Using `get_tool_async` (Inside Async Functions)
```python
from langroid.agent.tools.mcp.fastmcp_client import get_tool_async

async def main():
    transport = StdioTransport(command="...", args=[...])
    BaseTool = await get_tool_async(transport, "tool_name")
    
    class MyTool(BaseTool):
        async def handle_async(self):
            result = await self.call_tool_async()
            # custom processing
            return result
```

**Use this approach when**:
- Creating tools inside async functions
- Need to avoid event loop conflicts
- Want to delay transport creation until runtime

### Transport Types and Event Loop Considerations

- **StdioTransport**: Creates subprocess immediately, can cause "event loop closed" errors if created at module level in certain contexts
- **SSETransport**: HTTP-based, generally safer for module-level creation
- **Best Practice**: Create transports inside async functions when possible, use `asyncio.run()` wrapper for Fire CLI integration

### Tool Message Request Field and Agent Handlers

When you get an MCP tool named "my_tool", Langroid automatically:

1. **Sets the `request` field**: The dynamically created ToolMessage subclass has `request = "my_tool"`
2. **Enables custom agent handlers**: Agents can define these methods:
   - `my_tool()` - synchronous handler
   - `my_tool_async()` - async handler

The agent's message routing system automatically calls these handlers when the tool is used.

### Custom `handle_async` Method Override

Both decorator and non-decorator approaches support overriding `handle_async`:

```python
class MyTool(BaseTool):  # or use @mcp_tool decorator
    async def handle_async(self):
        # Get raw result from MCP server
        result = await self.call_tool_async()
        
        # Option 1: Return processed result to LLM (continues conversation)
        return f"<ProcessedResult>{result}</ProcessedResult>"
        
        # Option 2: Return ResultTool to terminate task
        return MyResultTool(answer=result)
```

### Common Async Issues and Solutions

**Problem**: "RuntimeError: asyncio.run() cannot be called from a running event loop"
**Solution**: Use `get_tool_async` instead of `@mcp_tool` decorator when already in async context

**Problem**: "RuntimeError: Event loop is closed"
**Solution**: 
- Move transport creation inside async functions
- Use `asyncio.run()` wrapper for Fire CLI integration:
```python
if __name__ == "__main__":
    import asyncio
    def run_main(**kwargs):
        asyncio.run(main(**kwargs))
    Fire(run_main)
```

### MCP Tool Integration Examples

See `examples/mcp/` for working examples:
- `gitmcp.py` - HTTP-based SSE transport
- `pyodide_code_executor.py` - Subprocess-based stdio transport with proper async handling

## Testing and Tool Message Patterns

### MockLM for Testing Tool Generation
- Use `MockLM` with `response_dict` to simulate LLM responses that include tool messages
- Set `tools=[ToolClass]` or `enable_message=[ToolClass]` on the agent to enable tool handling
- The `try_get_tool_messages()` method can extract tool messages from LLM responses with `all_tools=True`

### Task Termination Control
- `TaskConfig` has `done_if_tool` parameter to terminate tasks when any tool is generated
- `Task.done()` method checks `result.agent_response` for tool content when this flag is set
- Useful for workflows where tool generation signals task completion

### Testing Tool-Based Task Flows
```python
# Example: Test task termination on tool generation
config = TaskConfig(done_if_tool=True)
task = Task(agent, config=config)
response_dict = {"content": '{"request": "my_tool", "param": "value"}'}
```

## Commit and Pull Request Guidelines

- Never include "co-authored by Claude Code" or "created by Claude" in commit messages or pull request descriptions