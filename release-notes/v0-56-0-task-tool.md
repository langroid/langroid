# Release Notes: TaskTool

## New Feature: `TaskTool` for Spawning Sub-Agents

We've added `TaskTool`, a new tool that enables agents to spawn sub-agents for handling specific tasks. This allows for dynamic task delegation with controlled tool access.

### Key Features:
- Agents can spawn sub-agents with specific tools and configurations
- Sub-agents run non-interactively and return results to the parent
- Supports nested operations and recursive task delegation
- Flexible tool access: delegate specific tools, all tools, or no tools

### Example Usage:
```python
# Agent spawns a sub-agent to handle a calculation
{
    "request": "task_tool",
    "system_message": "You are a calculator. Use multiply_tool to compute products.",
    "prompt": "Calculate 5 * 7",
    "tools": ["multiply_tool"],
    "model": "gpt-4o-mini"
}
```

### Documentation:
Full documentation with examples: [TaskTool Documentation](https://langroid.github.io/langroid/notes/task-tool/)

### Testing:
See working examples in [`tests/main/test_task_tool.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_task_tool.py)

## Update v0.56.3: Agent Naming Support

### Enhancement:
- Added optional `agent_name` parameter to TaskTool
- Sub-agents can now be given custom names for better logging and debugging
- If not specified, auto-generates unique names in format `agent-{uuid}`

### Example:
```python
{
    "request": "task_tool",
    "system_message": "You are a calculator.",
    "prompt": "Calculate 5 * 7",
    "tools": ["multiply_tool"],
    "agent_name": "calculator-agent"  # Optional: custom name for logging
}
```