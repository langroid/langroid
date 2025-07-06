# Task Termination in Langroid

## Why Task Termination Matters

When building agent-based systems, one of the most critical yet challenging aspects is determining when a task should complete. Unlike traditional programs with clear exit points, agent conversations can meander, loop, or continue indefinitely. Getting termination wrong leads to two equally problematic scenarios:

**Terminating too early** means missing crucial information or cutting off an agent mid-process. Imagine an agent that searches for information, finds it, but terminates before it can process or summarize the results. The task completes "successfully" but fails to deliver value.

**Terminating too late** wastes computational resources, frustrates users, and can lead to repetitive loops where agents keep responding without making progress. We've all experienced chatbots that won't stop talking or systems that keep asking "Is there anything else?" long after the conversation should have ended. Even worse, agents can fall into infinite loops—repeatedly exchanging the same messages, calling the same tools, or cycling through states without making progress. These loops not only waste resources but can rack up significant costs when using paid LLM APIs.

The challenge is that the "right" termination point depends entirely on context. A customer service task might complete after resolving an issue and confirming satisfaction. A research task might need to gather multiple sources, synthesize them, and present findings. A calculation task should end after computing and presenting the result. Each scenario requires different termination logic.

Traditionally, developers would subclass `Task` and override the `done()` method with custom logic. While flexible, this approach scattered termination logic across multiple subclasses, making systems harder to understand and maintain. It also meant that common patterns—like "complete after tool use" or "stop when the user says goodbye"—had to be reimplemented repeatedly.

This guide introduces Langroid's declarative approach to task termination, culminating in the powerful `done_sequences` feature. Instead of writing imperative code, you can now describe *what* patterns should trigger completion, and Langroid handles the *how*. This makes your agent systems more predictable, maintainable, and easier to reason about.

## Table of Contents
- [Overview](#overview)
- [Basic Termination Methods](#basic-termination-methods)
- [Done Sequences: Event-Based Termination](#done-sequences-event-based-termination)
  - [Concept](#concept)
  - [DSL Syntax (Recommended)](#dsl-syntax-recommended)
  - [Full Object Syntax](#full-object-syntax)
  - [Event Types](#event-types)
  - [Examples](#examples)
- [Implementation Details](#implementation-details)
- [Best Practices](#best-practices)
- [Reference](#reference)

## Overview

In Langroid, a `Task` wraps an `Agent` and manages the conversation flow. Controlling when a task terminates is crucial for building reliable agent systems. Langroid provides several methods for task termination, from simple flags to sophisticated event sequence matching.

## Basic Termination Methods

### 1. Turn Limits
```python
# Task runs for exactly 5 turns
result = task.run("Start conversation", turns=5)
```

### 2. Single Round Mode
```python
# Task completes after one exchange
config = TaskConfig(single_round=True)
task = Task(agent, config=config)
```

### 3. Done If Tool
```python
# Task completes when any tool is generated
config = TaskConfig(done_if_tool=True)
task = Task(agent, config=config)
```

### 4. Done If Response/No Response
```python
# Task completes based on response from specific entities
config = TaskConfig(
    done_if_response=[Entity.LLM],      # Done if LLM responds
    done_if_no_response=[Entity.USER]   # Done if USER doesn't respond
)
```

### 5. String Signals
```python
# Task completes when special strings like "DONE" are detected
# (enabled by default with recognize_string_signals=True)
```

### 6. Orchestration Tools
```python
# Using DoneTool, FinalResultTool, etc.
from langroid.agent.tools.orchestration import DoneTool
agent.enable_message(DoneTool)
```

## Done Sequences: Event-Based Termination

### Concept

The `done_sequences` feature allows you to specify sequences of events that trigger task completion. This provides fine-grained control over task termination based on conversation patterns.

**Key Features:**

- Specify multiple termination sequences
- Use convenient DSL syntax or full object syntax
- Strict consecutive matching (no skipping events)
- Efficient implementation using message parent pointers

### DSL Syntax (Recommended)

The DSL (Domain Specific Language) provides a concise way to specify sequences:

```python
from langroid.agent.task import Task, TaskConfig

config = TaskConfig(
    done_sequences=[
        "T, A",                    # Tool followed by agent response
        "T[calculator], A",        # Specific calculator tool
        "L, T, A, L",              # LLM, tool, agent, LLM sequence
        "C[quit|exit|bye]",        # Content matching regex
        "U, L, A",                 # User, LLM, agent sequence
    ]
)
task = Task(agent, config=config)
```

#### DSL Pattern Reference

| Pattern | Description | Event Type |
|---------|-------------|------------|
| `T` | Any tool | `TOOL` |
| `T[name]` | Specific tool by name | `SPECIFIC_TOOL` |
| `A` | Agent response | `AGENT_RESPONSE` |
| `L` | LLM response | `LLM_RESPONSE` |
| `U` | User response | `USER_RESPONSE` |
| `N` | No response | `NO_RESPONSE` |
| `C[pattern]` | Content matching regex | `CONTENT_MATCH` |

**Examples:**

- `"T, A"` - Any tool followed by agent handling
- `"T[search], A, T[calculator], A"` - Search tool, then calculator tool
- `"L, C[complete|done|finished]"` - LLM response containing completion words
- `"TOOL, AGENT"` - Full words also supported

### Full Object Syntax

For more control, use the full object syntax:

```python
from langroid.agent.task import (
    Task, TaskConfig, DoneSequence, AgentEvent, EventType
)

config = TaskConfig(
    done_sequences=[
        DoneSequence(
            name="tool_handled",
            events=[
                AgentEvent(event_type=EventType.TOOL),
                AgentEvent(event_type=EventType.AGENT_RESPONSE),
            ]
        ),
        DoneSequence(
            name="specific_tool_pattern",
            events=[
                AgentEvent(
                    event_type=EventType.SPECIFIC_TOOL,
                    tool_name="calculator"
                ),
                AgentEvent(event_type=EventType.AGENT_RESPONSE),
            ]
        ),
    ]
)
```

### Event Types

The following event types are available:

| EventType | Description | Additional Parameters |
|-----------|-------------|----------------------|
| `TOOL` | Any tool message generated | - |
| `SPECIFIC_TOOL` | Specific tool by name | `tool_name` |
| `LLM_RESPONSE` | LLM generates a response | - |
| `AGENT_RESPONSE` | Agent responds (e.g., handles tool) | - |
| `USER_RESPONSE` | User provides input | - |
| `CONTENT_MATCH` | Response matches regex pattern | `content_pattern` |
| `NO_RESPONSE` | No valid response from entity | - |

### Examples

#### Example 1: Tool Completion
Task completes after any tool is used and handled:

```python
config = TaskConfig(done_sequences=["T, A"])
```

This is equivalent to `done_if_tool=True` but happens after the agent handles the tool.

#### Example 2: Multi-Step Process
Task completes after a specific conversation pattern:

```python
config = TaskConfig(
    done_sequences=["L, T[calculator], A, L"]
)
# Completes after: LLM response → calculator tool → agent handles → LLM summary
```

#### Example 3: Multiple Exit Conditions
Different ways to complete the task:

```python
config = TaskConfig(
    done_sequences=[
        "C[quit|exit|bye]",           # User says quit
        "T[calculator], A",           # Calculator used
        "T[search], A, T[search], A", # Two searches performed
    ]
)
```

#### Example 4: Mixed Syntax
Combine DSL strings and full objects:

```python
config = TaskConfig(
    done_sequences=[
        "T, A",  # Simple DSL
        DoneSequence(  # Full control
            name="complex_check",
            events=[
                AgentEvent(
                    event_type=EventType.SPECIFIC_TOOL,
                    tool_name="database_query",
                    responder="DatabaseAgent"  # Can specify responder
                ),
                AgentEvent(event_type=EventType.AGENT_RESPONSE),
            ]
        ),
    ]
)
```

## Implementation Details

### How Done Sequences Work

Done sequences operate at the **task level** and are based on the **sequence of valid responses** generated during a task's execution. When a task runs, it maintains a `response_sequence` that tracks each message (ChatDocument) as it's processed.

**Key points:**
- Done sequences are checked only within a single task's scope
- They track the temporal order of responses within that task
- The response sequence is built incrementally as the task processes each step
- Only messages that represent valid responses are added to the sequence

### Response Sequence Building
The task builds its response sequence during execution:

```python
# In task.run(), after each step:
if self.pending_message is not None:
    if (not self.response_sequence or 
        self.pending_message.id() != self.response_sequence[-1].id()):
        self.response_sequence.append(self.pending_message)
```

### Message Chain Retrieval
Done sequences are checked against the response sequence:

```python
def _get_message_chain(self, msg: ChatDocument, max_depth: Optional[int] = None):
    """Get the chain of messages from response sequence"""
    if max_depth is None:
        max_depth = 50  # default
        if self._parsed_done_sequences:
            max_depth = max(len(seq.events) for seq in self._parsed_done_sequences)
    
    # Simply return the last max_depth elements from response_sequence
    return self.response_sequence[-max_depth:]
```

**Note:** The response sequence used for done sequences is separate from the parent-child pointer system. Parent pointers track causal relationships and lineage across agent boundaries (important for debugging and understanding delegation patterns), while response sequences track temporal order within a single task for termination checking.

### Strict Matching
Events must occur consecutively without intervening messages:

```python
# This sequence: [TOOL, AGENT_RESPONSE]
# Matches: USER → LLM(tool) → AGENT
# Does NOT match: USER → LLM(tool) → USER → AGENT
```

### Performance

- Efficient O(n) traversal where n is sequence length
- No full history scan needed
- Early termination on first matching sequence

## Best Practices

1. **Use DSL for Simple Cases**
   ```python
   # Good: Clear and concise
   done_sequences=["T, A"]
   
   # Avoid: Verbose for simple patterns
   done_sequences=[DoneSequence(events=[...])]
   ```

2. **Name Your Sequences**
   ```python
   DoneSequence(
       name="calculation_complete",  # Helps with debugging
       events=[...]
   )
   ```

3. **Order Matters**
   - Put more specific sequences first
   - General patterns at the end

4. **Test Your Sequences**
   ```python
   # Use MockLM for testing
   agent = ChatAgent(
       ChatAgentConfig(
           llm=MockLMConfig(response_fn=lambda x: "test response")
       )
   )
   ```

5. **Combine with Other Methods**
   ```python
   config = TaskConfig(
       done_if_tool=True,      # Quick exit on any tool
       done_sequences=["L, L, L"],  # Or after 3 LLM responses
       max_turns=10,           # Hard limit
   )
   ```

## Reference

### Code Examples
- **Basic example**: [`examples/basic/done_sequences_example.py`](../../examples/basic/done_sequences_example.py)
- **Test cases**: [`tests/main/test_done_sequences.py`](../../tests/main/test_done_sequences.py)
- **DSL tests**: [`tests/main/test_done_sequences_dsl.py`](../../tests/main/test_done_sequences_dsl.py)
- **Parser tests**: [`tests/main/test_done_sequence_parser.py`](../../tests/main/test_done_sequence_parser.py)

### Core Classes
- `TaskConfig` - Configuration including `done_sequences`
- `DoneSequence` - Container for event sequences
- `AgentEvent` - Individual event in a sequence
- `EventType` - Enumeration of event types

### Parser Module
- `langroid.agent.done_sequence_parser` - DSL parsing functionality

### Task Methods
- `Task.done()` - Main method that checks sequences
- `Task._matches_sequence_with_current()` - Sequence matching logic
- `Task._classify_event()` - Event classification
- `Task._get_message_chain()` - Message traversal

## Migration Guide

If you're currently overriding `Task.done()`:

```python
# Before: Custom done() method
class MyTask(Task):
    def done(self, result=None, r=None):
        if some_complex_logic(result):
            return (True, StatusCode.DONE)
        return super().done(result, r)

# After: Use done_sequences
config = TaskConfig(
    done_sequences=["T[my_tool], A, L"]  # Express as sequence
)
task = Task(agent, config=config)  # No subclassing needed
```

## Troubleshooting

**Sequence not matching?**

- Check that events are truly consecutive (no intervening messages)
- Use logging to see the actual message chain
- Verify tool names match exactly

**Type errors with DSL?**

- Ensure you're using strings for DSL patterns
- Check that tool names in `T[name]` don't contain special characters

**Performance concerns?**

- Sequences only traverse as deep as needed
- Consider shorter sequences for better performance
- Use specific tool names to avoid unnecessary checks

## Summary

The `done_sequences` feature provides a powerful, declarative way to control task termination based on conversation patterns. The DSL syntax makes common cases simple while the full object syntax provides complete control when needed. This approach eliminates the need to subclass `Task` and override `done()` for most use cases, leading to cleaner, more maintainable code.