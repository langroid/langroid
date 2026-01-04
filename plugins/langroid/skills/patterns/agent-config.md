# Pattern: Basic Agent Configuration

## Problem

You need to create a Langroid ChatAgent with proper LLM configuration,
system message, and behavior settings.

## Solution

Use `ChatAgentConfig` to configure the agent, then instantiate `ChatAgent`.

## Complete Code Example

```python
import langroid as lr
from langroid.language_models import OpenAIGPTConfig

# Configure the agent
config = lr.ChatAgentConfig(
    name="ResearchAgent",
    llm=OpenAIGPTConfig(
        chat_model="gpt-4o",
        chat_context_length=128_000,
        temperature=0.7,
        timeout=120,
    ),
    system_message="""
    You are a research assistant. Help users find and analyze information.
    Be thorough and cite your sources.
    """,
    vecdb=None,  # No vector database for this agent
)

# Create the agent
agent = lr.ChatAgent(config)

# Wrap in a task and run
task = lr.Task(agent, interactive=True)
task.run()
```

## Key Configuration Options

| Option | Description |
|--------|-------------|
| `name` | Agent identifier (used in logs and multi-agent routing) |
| `llm` | LLM configuration (model, temperature, timeout, etc.) |
| `system_message` | Instructions for the agent's behavior |
| `vecdb` | Vector database config (None if not needed) |
| `handle_llm_no_tool` | What to do when LLM doesn't use a tool |

## Common LLM Settings

```python
llm=OpenAIGPTConfig(
    chat_model="gpt-4o",           # Model name
    chat_context_length=128_000,   # Context window size
    temperature=0.7,               # Creativity (0.0-1.0)
    timeout=120,                   # API timeout in seconds
    max_output_tokens=4096,        # Max response length
)
```

## When to Use

- Starting point for any Langroid agent
- Simple single-agent applications
- Building blocks for multi-agent systems
