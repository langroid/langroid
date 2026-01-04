# Pattern: Task Termination with done_sequences

## Problem

You need fine-grained control over when a task terminates - specific tools,
sequences of events, or content patterns.

## Solution

Use `TaskConfig(done_sequences=[...])` with DSL patterns or
`done_if_tool=True` for simpler cases.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class SearchTool(ToolMessage):
    """Intermediate tool - should NOT terminate."""
    request: str = "search"
    purpose: str = "Search for information"
    query: str = Field(..., description="Search query")


class FinalAnswerTool(ToolMessage):
    """Final tool - SHOULD terminate."""
    request: str = "final_answer"
    purpose: str = "Provide the final answer"
    answer: str = Field(..., description="The answer")
    confidence: float = Field(..., description="0.0-1.0")


agent = lr.ChatAgent(lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    system_message="Use search to find info, then final_answer when done.",
))
agent.enable_message([SearchTool, FinalAnswerTool])


# Option 1: Terminate on ANY tool
task = Task(
    agent,
    interactive=False,
    config=TaskConfig(done_if_tool=True),
)


# Option 2: Terminate on SPECIFIC tool (emission only)
task = Task(
    agent,
    interactive=False,
    config=TaskConfig(done_sequences=["T[FinalAnswerTool]"]),
)[FinalAnswerTool]


# Option 3: Terminate after tool is HANDLED
task = Task(
    agent,
    interactive=False,
    config=TaskConfig(done_sequences=["T[FinalAnswerTool], A"]),
)[FinalAnswerTool]


# Run and get typed result
result: FinalAnswerTool | None = task.run("What is the capital of France?")
if result:
    print(f"Answer: {result.answer} (confidence: {result.confidence})")
```

## DSL Syntax Reference

| Pattern | Meaning |
|---------|---------|
| `T` | Any tool emitted |
| `T[ToolName]` | Specific tool by class name |
| `A` | Agent response (handler ran) |
| `L` | LLM response |
| `C[pattern]` | Content matches regex pattern |
| `,` | Then (sequence of events) |

## Common Patterns

```python
# Terminate on any tool
done_sequences=["T"]

# Terminate on specific tool
done_sequences=["T[FinalAnswerTool]"]

# Terminate after tool + handler
done_sequences=["T[FinalAnswerTool], A"]

# Multiple exit conditions (OR logic)
done_sequences=[
    "T[FinalAnswerTool]",  # Exit on this tool
    "C[quit|exit|bye]",    # OR user says quit
]

# Complex sequence
done_sequences=["L, T[SearchTool], A, T[AnswerTool], A"]
```

## Emission vs Handling

| Pattern | When it exits |
|---------|---------------|
| `T[Tool]` | Immediately when LLM emits tool |
| `T[Tool], A` | After tool is emitted AND handled |

Use `, A` when:
- Handler has side effects that must complete
- Handler validates and may return error (causing retry)
- You want handler's return value in result

## Key Points

- `done_if_tool=True` is simpler but less flexible
- Use `done_sequences` for specific tool termination
- Combine with `[ToolType]` subscript for typed return
- Multiple patterns in list = OR logic (any can terminate)
- Order matters in sequences: events must occur in that order

## When to Use

- Agent has multiple tools, only one should terminate
- Complex workflows with specific termination conditions
- Content-based termination (user says "quit")
- Ensuring handler runs before termination
