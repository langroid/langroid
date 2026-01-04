---
name: patterns
description: Design patterns for the Langroid multi-agent LLM framework. Covers
  agent configuration, tools, task control, and multi-agent orchestration.
---

# Langroid Patterns

## Instructions

Below is an INDEX of Langroid design patterns. Each item describes WHAT you might
want to implement, followed by a POINTER to a document with a complete code example.

Scan this index to find patterns that match your needs, then consult the
corresponding document.

---

## Agent & Configuration Patterns

1. **Basic Agent Configuration** - Set up a ChatAgent with LLM config,
   system message, and basic settings. Use this as the foundation for any
   Langroid agent.
   see ./agent-config.md

2. **Custom Config Subclass** - Create a reusable ChatAgentConfig subclass
   with preset fields (name, system_message, llm settings). Use this when you
   have multiple agents sharing similar configuration.
   see ./custom-agent-config.md

3. **Agent with Custom State** - Create a ChatAgent subclass with custom
   `__init__` that stores state (connections, counters, context). Use this when
   tool handlers need access to agent-level data or resources.
   see ./agent-state.md

---

## Tool Patterns

4. **Basic ToolMessage Definition** - Define a tool with `request`, `purpose`,
   and Pydantic `Field()` descriptions. Use this for any structured output you
   want the LLM to generate.
   see ./tool-message-basic.md

5. **Tool Examples for Few-Shot** - Add an `examples()` classmethod returning
   sample tool instances. Use this to improve LLM tool usage accuracy through
   few-shot learning.
   see ./tool-examples.md

6. **Stateless Tool Handler (in ToolMessage)** - Define `handle()` method
   inside the ToolMessage class. Use this for simple validation or
   transformation that doesn't need agent state.
   see ./tool-handle-stateless.md

7. **Stateful Tool Handler (on Agent)** - Define handler method on agent class
   with name matching tool's `request` field. Use this when the handler needs
   agent state, external resources, or complex logic.
   see ./agent-handler-stateful.md

8. **Enabling Tools with enable_message()** - Register tools on an agent using
   `enable_message()` with single tool, list of tools, or use/handle flags.
   see ./enable-message.md

---

## Task Control Patterns

9. **Task Termination with done_sequences** - Control when a task terminates
   using `TaskConfig(done_sequences=...)` or `done_if_tool=True`. Use this for
   fine-grained control over task completion.
   see ./done-sequences.md

10. **Typed Task Return with Subscript** - Use `Task(...)[ToolType]` to get
    typed tool output directly. Use this when you want the task to return a
    specific tool instance rather than ChatDocument.
    see ./task-subscript.md

11. **Sequential Multi-Agent Orchestration** - Chain multiple agents by
    running tasks sequentially, passing output from one to the next. Use this
    for workflows like Writer → Reviewer → Editor.
    see ./sequential-orchestration.md

---

## Integration Patterns

12. **MCP Tool Integration** - Connect to MCP servers (like Claude Code) and
    enable their tools on Langroid agents. Use `@mcp_tool` decorator or
    `get_tools_async()` for all tools.
    see ./mcp-tools.md

13. **Quiet Mode for Clean Output** - Suppress verbose Langroid output using
    `quiet_mode()` context manager. Use this when you want clean CLI output
    with only your custom progress messages.
    see ./quiet-mode.md

14. **Force Tool Usage with handle_llm_no_tool** - Configure agent to return
    an error message when LLM generates plain text instead of using a tool.
    Use this to enforce tool-only output.
    see ./handle-llm-no-tool.md
