---
name: patterns
description: Design patterns for the Langroid multi-agent LLM framework. Covers
  agent configuration, tools, task control, and integrations.
---

# Langroid Patterns

## Instructions

Below is an INDEX of design patterns organized by category. Each item describes
WHAT you might want to implement, followed by a REFERENCE to a document with
a complete code example.

Scan this index to find patterns matching your needs, then consult the
corresponding document.

---

## Agent & Task Basics

1. **Task Returns Tool Directly**

   Create a Langroid Agent equipped with a single Tool (a ToolMessage), and wrap
   it in a Task so that running the task returns that ToolMessage directly. Use
   this pattern when you want a simple LLM agent that returns a structured
   response.

   - Reference: `./task-return-tool.md`

---

## Tool Handlers

2. **Stateful Handler on Agent**

   Define a STATEFUL tool handler as a METHOD on the agent (not inside the
   ToolMessage). Use this pattern when: (a) the tool handler needs to execute
   external operations (API calls, database queries, file I/O), (b) you need to
   track state across retries (e.g., failure counter), (c) the handler needs
   access to agent-level resources (connections, configs), or (d) you want
   Langroid to automatically loop errors back to the LLM for self-correction.
   The method name must match the `request` field of the ToolMessage. Return a
   string for errors (LLM sees it and can retry), or DoneTool(content=result)
   to terminate successfully.

   - Reference: `./agent-tool-handler-with-state.md`

3. **Handler with Validation**

   Validate tool output against agent state before accepting it. Use this
   pattern when: (a) the LLM's tool output must preserve certain content from
   the input (e.g., placeholders, required fields), (b) you want automatic
   retry if validation fails, (c) you need to compare tool output against
   context the LLM received. Define a handler method on a custom agent class
   that stores the input context as state, validates the tool output, and
   returns an error string for retry or AgentDoneTool for success (note: use
   AgentDoneTool, NOT DoneTool). Use `done_sequences=["T[ToolName], A"]` so the
   handler runs before task termination.

   - Reference: `./agent-handler-validation-with-state.md`

---

## Task Control

4. **Terminate on Specific Tool**

   Terminate a Task only when a SPECIFIC tool is called. Use
   `TaskConfig(done_sequences=["T[ToolName]"])` to exit immediately when that
   tool is emitted, or `TaskConfig(done_sequences=["T[ToolName], A"])` to exit
   after the tool is emitted AND handled by the agent. Use this when an agent
   has multiple tools but you only want one specific tool to trigger task
   termination.

   - Reference: `./done-sequences-specific-tool.md`

5. **Batch Processing**

   Run the SAME task on MULTIPLE inputs concurrently using `run_batch_tasks()`.
   Use this pattern when: (a) you need to process many items with the same
   agent/task logic, (b) you want parallelism without manual asyncio/threading,
   (c) you need state isolation between items (each gets a cloned agent with
   fresh message history), (d) you want to avoid connection exhaustion from
   creating too many agents manually. Each item gets a cloned task+agent, runs
   independently, results collected in order. Supports batch_size for
   concurrency limiting.

   - Reference: `./run-batch-tasks.md`

---

## Integration & Output

6. **MCP Tools Integration**

   Enable a Langroid agent to use MCP (Model Context Protocol) tools from an
   external MCP server like Claude Code. Use this pattern when: (a) you want
   your agent to use file editing tools (Read, Edit, Write) from Claude Code,
   (b) you need to connect to any MCP server via stdio transport, (c) you want
   to enable ALL tools from an MCP server or just SPECIFIC tools selectively,
   (d) you want to customize/post-process MCP tool results before returning to
   the LLM. Uses `@mcp_tool` decorator for specific tools or `get_tools_async()`
   for all tools.

   - Reference: `./mcp-tool-integration.md`

7. **Quiet Mode**

   Suppress verbose Langroid agent output (streaming, tool JSON, intermediate
   messages) while showing your own custom progress messages. Use this pattern
   when: (a) you want clean CLI output showing only milestone events, (b) you're
   running a multi-step workflow and want to show progress without agent noise,
   (c) you need thread-safe output control. Use `quiet_mode()` context manager
   to wrap agent task.run() calls, then print your own messages outside the
   context.

   - Reference: `./quiet-mode.md`
