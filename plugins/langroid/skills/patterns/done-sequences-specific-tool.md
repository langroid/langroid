# Pattern: Terminate Task on SPECIFIC Tool (done_sequences)

## Problem

You have an agent with multiple tools, but you only want the task to terminate
when ONE specific tool is called. Other tools should NOT trigger termination.

## Solution

Use `TaskConfig(done_sequences=["T[ToolName]"])` with the specific tool name.

### Two Variants

**Exit immediately on tool EMISSION:**
```python
task_config = lr.TaskConfig(
    done_sequences=["T[FinalAnswerTool]"]  # No ", A"
)
```
Task terminates as soon as the LLM emits `FinalAnswerTool`, before any handling.

**Exit after tool is HANDLED:**
```python
task_config = lr.TaskConfig(
    done_sequences=["T[FinalAnswerTool], A"]  # With ", A"
)
```
Task waits for the tool to be emitted AND for the agent to handle it before
terminating.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig


class SearchTool(ToolMessage):
    """Intermediate tool - should NOT trigger exit."""
    request: str = "search"
    purpose: str = "Search for information"
    query: str


class FinalAnswerTool(ToolMessage):
    """Final tool - SHOULD trigger exit."""
    request: str = "final_answer"
    purpose: str = "Provide the final answer"
    answer: str
    confidence: float


def create_agent() -> ChatAgent:
    config = ChatAgentConfig(
        name="ResearchAgent",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="""
You are a research agent. Use the search tool to find information,
then use final_answer when you have enough to answer confidently.
""",
    )
    agent = ChatAgent(config)
    agent.enable_message(SearchTool)
    agent.enable_message(FinalAnswerTool)
    return agent


def research(question: str) -> str | None:
    agent = create_agent()

    # Only exit when FinalAnswerTool is used (SearchTool won't trigger exit)
    task_config = lr.TaskConfig(
        done_sequences=["T[FinalAnswerTool]"]
    )
    task = Task(agent, interactive=False, config=task_config)[FinalAnswerTool]

    # Agent can use SearchTool multiple times without exiting
    # Task only exits when FinalAnswerTool is emitted
    result: FinalAnswerTool | None = task.run(question, turns=15)

    if result:
        return result.answer
    return None
```

## DSL Syntax Reference

| Pattern | Meaning |
|---------|---------|
| `T` | Any tool |
| `T[ToolName]` | Specific tool by class name |
| `A` | Agent response (tool handling) |
| `C[pattern]` | Content matching regex pattern |
| `,` | Then (sequence of events) |

## Key Differences Between Variants

| Pattern | When it exits | Use case |
|---------|---------------|----------|
| `["T[Tool]"]` | Immediately on emission | Get tool output, no handling needed |
| `["T[Tool], A"]` | After emission + handling | Tool has side effects to complete |

## Complex Patterns

### Exit after two specific tools in sequence
```python
done_sequences=["T[SearchTool], A, T[AnalyzeTool], A"]
```

### Multiple exit conditions (OR logic)
```python
done_sequences=[
    "C[quit|exit|bye]",      # Exit if user says quit
    "T[FinalAnswerTool]"     # OR if FinalAnswerTool is used
]
```

### Exit only after tool AND specific content
```python
done_sequences=["T[CompletionTool], A, C[done|complete]"]
```

## When to Use This Pattern

- Agent has multiple tools but only ONE should trigger exit
- Other tools are intermediate steps that should NOT terminate the task
- You need fine-grained control over which tool ends the conversation

## Common Mistake

```python
# WRONG: Bracket notation does NOT filter which tools trigger exit
# It only specifies the RETURN TYPE
task = Task(agent, config=task_config)[FinalAnswerTool]
```

The bracket notation `[FinalAnswerTool]` specifies what type the task returns.
To control which tool TRIGGERS exit, you must use `done_sequences`.
