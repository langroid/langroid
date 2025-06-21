# TaskTool: Spawning Sub-Agents for Task Delegation

## Overview

`TaskTool` allows agents to **spawn sub-agents** to handle specific tasks. When an agent encounters a task that requires specialized tools or isolated execution, it can spawn a new sub-agent with exactly the capabilities needed for that task.

This enables agents to dynamically create a hierarchy of specialized workers, each focused on their specific subtask with only the tools they need.

## When to Use TaskTool

TaskTool is useful when:
- Different parts of a task require different specialized tools
- You want to isolate tool access for specific operations  
- A task involves recursive or nested operations
- You need different LLM models for different subtasks

## How It Works

1. The parent agent decides to spawn a sub-agent and specifies:
   - A system message defining the sub-agent's role
   - A prompt for the sub-agent to process
   - Which tools the sub-agent should have access to
   - Optional model and iteration limits

2. TaskTool spawns the new sub-agent, runs the task, and returns the result to the parent.

## Async Support

TaskTool fully supports both synchronous and asynchronous execution. The tool automatically handles async contexts when the parent task is running asynchronously.

## Usage Example

```python
from langroid.agent.tools.task_tool import TaskTool

# Enable TaskTool for your agent
agent.enable_message([TaskTool, YourCustomTool], use=True, handle=True)

# Agent can now spawn sub-agents for tasks when the LLM generates a task_tool request:

response = {
    "request": "task_tool",
    "system_message": "You are a calculator. Use the multiply_tool to compute products.",
    "prompt": "Calculate 5 * 7",
    "tools": ["multiply_tool"],
    "model": "gpt-4o-mini",  # optional
    "max_iterations": 5       # optional
}
```

## Field Reference

**Required fields:**
- `system_message`: Instructions for the sub-agent's role and behavior
- `prompt`: The specific task/question for the sub-agent
- `tools`: List of tool names. Special values: `["ALL"]` or `["NONE"]`

**Optional fields:**
- `model`: LLM model name (default: "gpt-4o-mini")
- `max_iterations`: Task iteration limit (default: 10)

## Example: Nested Operations

Consider computing `Nebrowski(10, Nebrowski(3, 2))` where Nebrowski is a custom operation. The main agent spawns sub-agents to handle each operation:

```python
# Main agent spawns first sub-agent for inner operation:
{
    "request": "task_tool",
    "system_message": "Compute Nebrowski operations using the nebrowski_tool.",
    "prompt": "Compute Nebrowski(3, 2)",
    "tools": ["nebrowski_tool"]
}

# Then spawns another sub-agent for outer operation:
{
    "request": "task_tool",
    "system_message": "Compute Nebrowski operations using the nebrowski_tool.",
    "prompt": "Compute Nebrowski(10, 11)",  # where 11 is the previous result
    "tools": ["nebrowski_tool"]
}
```

## Working Examples

See [`tests/main/test_task_tool.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_task_tool.py) for complete examples including:
- Basic task delegation with mock agents
- Nested operations with custom tools
- Both sync and async usage patterns

## Important Notes

- Spawned sub-agents run non-interactively (no human input)
- `DoneTool` is automatically enabled for all sub-agents
- Results are returned as `ChatDocument` objects. The Langroid framework takes care
  of converting them to a suitable format for the parent agent's LLM to consume and 
  respond to.
- Only tools "known" to the parent agent can be enabled for sub-agents. This is an 
  important aspect of the current mechanism. The `TaskTool` handler method in
  the sub-agent only has access to tools that are known to the parent agent.
  If there are tools that are only relevant to the sub-agent but not the parent,
  you must still enable them in the parent agent, but you can set `use=False`
  and `handle=False` when you enable them, e.g.:

```python
agent.enable_message(MySubAgentTool, use=False, handle=False)
```
  Since we are letting the main agent's LLM "decide" when to spawn a sub-agent,
  your system message of the main agent should contain instructions clarifying that
  it can decide which tools to enable for the sub-agent, as well as a list of 
  all tools that might possibly be relevant to the sub-agent. This is particularly
  important for tools that have been enabled with `use=False`, since instructions for
  such tools would not be auto-inserted into the agent's system message. 



## Best Practices

1. **Clear Instructions**: Provide specific system messages that explain the sub-agent's role and tool usage
2. **Tool Availability**: Ensure delegated tools are enabled for the parent agent
3. **Appropriate Models**: Use simpler/faster models for simple subtasks
4. **Iteration Limits**: Set reasonable limits based on task complexity