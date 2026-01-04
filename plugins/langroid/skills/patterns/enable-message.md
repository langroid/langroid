# Pattern: Enabling Tools with enable_message()

## Problem

You need to register tools on an agent so the LLM knows about them and
handlers are connected.

## Solution

Use `agent.enable_message()` with a single tool, list of tools, or with
`use`/`handle` flags for fine-grained control.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class SearchTool(ToolMessage):
    request: str = "search"
    purpose: str = "Search for information"
    query: str = Field(..., description="Search query")


class AnalyzeTool(ToolMessage):
    request: str = "analyze"
    purpose: str = "Analyze data"
    data: str = Field(..., description="Data to analyze")


class ResultTool(ToolMessage):
    request: str = "result"
    purpose: str = "Return final result"
    answer: str = Field(..., description="The answer")


agent = lr.ChatAgent(lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    system_message="You are a research assistant.",
))


# Method 1: Single tool
agent.enable_message(SearchTool)


# Method 2: List of tools
agent.enable_message([SearchTool, AnalyzeTool, ResultTool])


# Method 3: With use/handle flags
agent.enable_message(SearchTool, use=True, handle=True)    # LLM can emit, agent handles
agent.enable_message(AnalyzeTool, use=True, handle=False)  # LLM can emit, no handler
agent.enable_message(ResultTool, use=False, handle=True)   # LLM won't emit, agent handles
```

## Flag Combinations

| `use` | `handle` | Effect |
|-------|----------|--------|
| `True` | `True` | LLM can emit tool, agent handles it (default) |
| `True` | `False` | LLM can emit tool, no automatic handling |
| `False` | `True` | Tool not shown to LLM, but handler exists |
| `False` | `False` | Tool disabled entirely |

## Common Patterns

### Enable all tools with default handling
```python
agent.enable_message([Tool1, Tool2, Tool3])
```

### Tools for LLM output only (no handler needed)
```python
# LLM emits these, we just extract the data
agent.enable_message(OutputTool, use=True, handle=False)
```

### Internal tools (agent handles, LLM doesn't use)
```python
# Handler receives messages from other agents
agent.enable_message(InternalTool, use=False, handle=True)
```

### Conditional tool enabling
```python
def create_agent(enable_search: bool = True) -> lr.ChatAgent:
    agent = lr.ChatAgent(config)

    tools = [ResultTool]  # Always enabled
    if enable_search:
        tools.append(SearchTool)

    agent.enable_message(tools)
    return agent
```

## Key Points

- Call `enable_message()` AFTER creating agent, BEFORE running task
- Pass list for multiple tools: `[Tool1, Tool2]`
- Default is `use=True, handle=True`
- Tools must be enabled for LLM to know about them
- Handler methods must exist on agent if `handle=True`

## When to Use

- Always - tools must be enabled to work
- Use flags when tools serve different purposes
- Use lists to enable multiple tools at once
