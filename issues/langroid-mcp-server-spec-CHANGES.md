# Langroid MCP Server Implementation: Strategy and Changes

## Overview

This document summarizes the implementation strategy and changes made to create an MCP (Model Context Protocol) server that exposes Langroid agents as tools, enabling any MCP client to use Langroid's agent framework with their own LLM through MCP's sampling feature.

## Implementation Strategy

### Core Innovation: ClientLM

The key innovation is the `ClientLM` class - a new LanguageModel implementation that delegates LLM calls back to the MCP client via the MCP context's sampling handler. This decouples Langroid agents from specific LLM providers, allowing them to use whatever LLM the MCP client provides.

### Architecture Components

1. **ClientLM (langroid/language_models/client_lm.py)**
   - Subclass of LanguageModel that uses MCP context for LLM operations
   - Converts between Langroid's message format and MCP's format
   - Supports async operations only (as required by MCP)

2. **MCP Server (langroid/mcp/server/langroid_mcp_server.py)**
   - FastMCP-based server exposing two tools:
     - `langroid_chat`: Single-turn chat with a Langroid agent
     - `langroid_task`: Multi-turn task execution with a Langroid agent
   - Both tools accept MCP context and create agents using ClientLM

3. **Integration with Langroid Factory Pattern**
   - Modified LanguageModel.create() to support "client" type
   - Seamless integration with existing Langroid infrastructure

## Key Changes Made

### 1. New Files Created

#### langroid/language_models/client_lm.py
```python
class ClientLMConfig(LLMConfig):
    type: str = "client"
    context: Optional[Any] = None  # MCP context set at runtime
    chat_context_length: int = 1_000_000_000  # effectively infinite

class ClientLM(LanguageModel):
    # Async-only implementation that delegates to MCP context.sample()
    # Handles message format conversion between Langroid and MCP
```

#### langroid/mcp/server/langroid_mcp_server.py
```python
@server.tool()
async def langroid_chat(message: str, ctx: Context, ...):
    # Single-turn chat with Langroid agent

@server.tool()
async def langroid_task(message: str, ctx: Context, ...):
    # Multi-turn task execution
```

### 2. Modified Files

#### langroid/language_models/base.py
- Added support for "client" type in LanguageModel.create() factory method:
```python
if config.type == "client":
    return ClientLM(cast(ClientLMConfig, config))
```

#### pyproject.toml
- Added script entry point for UVX installation:
```toml
[project.scripts]
langroid-mcp-server = "langroid.mcp.server.langroid_mcp_server:main"
```

### 3. Test Suite Created

#### tests/main/test_client_lm.py
- Comprehensive unit tests for ClientLM (10 tests)
- Tests async operations, message conversion, error handling
- Uses mock MCP context to simulate sampling behavior

#### tests/main/test_langroid_mcp_server.py
- Server tests using real Langroid components (not mocks)
- Tests both langroid_chat and langroid_task tools
- Verifies tool enablement and response handling

#### tests/main/test_mcp_langroid_integration.py
- End-to-end integration tests
- Tests multi-turn conversations, tool usage, error scenarios
- Verifies message format conversion and context preservation

### 4. Documentation and Examples

#### examples/mcp/langroid_agent_client.py
- Complete client example showing both OpenAI and Anthropic sampling handlers
- Demonstrates single-turn chat, multi-turn tasks, and tool usage
- Includes both stdio and SSE transport examples

#### examples/mcp/README.md
- Comprehensive user documentation
- Installation instructions for both server and client
- Usage examples and troubleshooting guide

## Technical Decisions

### 1. Async-Only Implementation
- ClientLM only supports async operations (achat, agenerate)
- Synchronous methods raise NotImplementedError
- This aligns with MCP's async-first design

### 2. Message Format Conversion
- System messages handled separately (MCP expects system_prompt parameter)
- Role mapping between Langroid and MCP formats
- Graceful handling of different response formats from MCP

### 3. Context Management
- MCP context passed via ClientLMConfig at runtime
- No global state or module-level transport creation
- Clean separation of concerns

### 4. Error Handling
- RuntimeError when no MCP context available
- Proper error propagation from MCP sampling
- Graceful fallbacks for response format variations

### 5. Testing Strategy
- Used real components instead of heavy mocking (per user preference)
- Created MockContext that simulates MCP behavior
- Set interactive=False for Task creation to avoid input prompts

## Installation and Usage

### Server Installation
```bash
# Install directly with uvx (recommended)
uvx langroid-mcp-server

# Or install in environment
pip install langroid
langroid-mcp-server
```

### Client Usage
```python
# Create transport and connect to server
transport = StdioTransport(command="langroid-mcp-server")
mcp_client = MCPClient(transport)
await mcp_client.connect()

# List available tools
tools = await mcp_client.list_tools()

# Use langroid_chat tool
result = await mcp_client.call_tool("langroid_chat", {
    "message": "Hello, how are you?",
    "enable_tools": ["web_search"]
})
```

## Benefits Achieved

1. **Decoupling**: Langroid agents no longer tied to specific LLM providers
2. **Flexibility**: Any MCP client can use Langroid's agent framework
3. **Integration**: Seamless integration with existing Langroid codebase
4. **Standards Compliance**: Full MCP protocol compliance
5. **Tool Support**: Langroid tools accessible via MCP interface
6. **Easy Installation**: Simple UVX-based deployment

## Future Enhancements

1. Support for more Langroid tools in the MCP interface
2. Streaming support when MCP protocol adds it
3. Additional configuration options for agents
4. Performance optimizations for high-throughput scenarios