# Langroid MCP Server Examples

This directory contains examples demonstrating how to use the Langroid MCP (Model Context Protocol) server, which allows any MCP client to use Langroid agents with their own LLM through MCP's sampling feature.

## Overview

The Langroid MCP server exposes Langroid agents as MCP tools, enabling:
- Use of Langroid's powerful agent framework from any MCP client
- Complete decoupling between Langroid agents and specific LLM providers
- Client control over which LLM to use through MCP sampling

## Architecture

```
MCP Client (with LLM) <-> Langroid MCP Server <-> Langroid Agent (with ClientLM)
```

The key innovation is that the Langroid agent uses a special `ClientLM` that delegates all LLM calls back to the MCP client via the sampling protocol.

## Installation

The Langroid MCP server can be installed and run via UVX:

```bash
uvx --from langroid langroid-mcp-server
```

## Available Tools

The server exposes two main tools:

### 1. `langroid_chat`
Simple chat interface with a Langroid agent.

Parameters:
- `message` (str): User message to send to agent
- `enable_tools` (List[str], optional): List of tools to enable (e.g., ["web_search"])
- `agent_name` (str, optional): Custom name for the agent

### 2. `langroid_task`
Task-based interface for multi-turn conversations.

Parameters:
- `message` (str): Initial user message
- `enable_tools` (List[str], optional): List of tools to enable
- `agent_name` (str, optional): Custom name for the agent
- `max_turns` (int): Maximum conversation turns (default: 10)

## Client Example

See `langroid_agent_client.py` for a complete example that shows:

1. **Basic Setup**: How to connect to the Langroid MCP server
2. **Sampling Handler**: Implementation for different LLM providers (OpenAI, Anthropic)
3. **Simple Chat**: Basic question-answering with Langroid agents
4. **Tool Usage**: Enabling web search and other tools
5. **Task-based Interaction**: Multi-turn conversations for complex tasks

### Running the Example

1. Start the Langroid MCP server:
   ```bash
   uvx --from langroid langroid-mcp-server
   ```

2. Set your LLM API keys:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export ANTHROPIC_API_KEY="your-key-here"
   export LLM_PROVIDER="anthropic"
   ```

3. Run the client example:
   ```bash
   python langroid_agent_client.py
   ```

## Custom Sampling Handlers

You can implement custom sampling handlers for any LLM provider. The handler should:

1. Accept MCP messages and sampling parameters
2. Convert to your LLM's format
3. Call your LLM API
4. Return the response text

Example structure:
```python
async def custom_sampling_handler(
    messages: List[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    # Convert messages to your LLM format
    # Call your LLM
    # Return response text
```

## Available Langroid Tools

Currently supported tools that can be enabled:
- `web_search`: DuckDuckGo web search capability

More tools can be added by extending the server implementation.

## Use Cases

1. **LLM Flexibility**: Use Langroid agents with any LLM (OpenAI, Anthropic, local models, etc.)
2. **Cost Optimization**: Switch between different LLM providers based on task requirements
3. **Privacy**: Use local LLMs with Langroid's agent capabilities
4. **Experimentation**: Easily test different LLMs with the same agent logic

## Troubleshooting

### Server Won't Start
- Ensure you have the latest version of Langroid installed
- Check that no other process is using the MCP server ports

### Connection Issues
- Verify the server is running before starting the client
- Check that your environment has the required dependencies

### LLM Errors
- Ensure your API keys are correctly set
- Check that you have credits/quota with your LLM provider
- Verify network connectivity

## Further Reading

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Langroid Documentation](https://langroid.github.io/langroid/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)