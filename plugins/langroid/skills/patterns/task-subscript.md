# Pattern: Typed Task Return with Subscript

## Problem

You want `task.run()` to return a specific tool type directly, rather than
a `ChatDocument` that you have to extract the tool from.

## Solution

Use `Task(...)[ToolType]` subscript notation. The task will return an
instance of `ToolType` (or `None`) directly.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class AnalysisResult(ToolMessage):
    """Structured analysis output."""
    request: str = "analysis_result"
    purpose: str = "Return the analysis result"

    summary: str = Field(..., description="Summary of analysis")
    score: float = Field(..., description="Score from 0-100")
    recommendations: list[str] = Field(..., description="List of recommendations")


def analyze_text(text: str) -> AnalysisResult | None:
    """Analyze text and return structured result."""

    agent = lr.ChatAgent(lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="""
        Analyze the provided text and return your analysis using
        the analysis_result tool.
        """,
    ))
    agent.enable_message(AnalysisResult)

    # Subscript notation: Task[ToolType]
    task = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )[AnalysisResult]  # <-- This makes run() return AnalysisResult | None

    # result is typed as AnalysisResult | None
    result: AnalysisResult | None = task.run(text)

    return result


# Usage
analysis = analyze_text("This product has great features but poor documentation.")

if analysis:
    print(f"Summary: {analysis.summary}")
    print(f"Score: {analysis.score}")
    for rec in analysis.recommendations:
        print(f"- {rec}")
else:
    print("Analysis failed")
```

## Without vs With Subscript

```python
# WITHOUT subscript - returns ChatDocument
task = Task(agent, interactive=False)
result = task.run(prompt)  # result is ChatDocument
# Must extract tool manually:
tool = result.tool_messages[0] if result.tool_messages else None


# WITH subscript - returns ToolType directly
task = Task(agent, interactive=False)[AnalysisResult]
result = task.run(prompt)  # result is AnalysisResult | None
# Use directly:
if result:
    print(result.summary)
```

## Combining with done_sequences

```python
# Both specific termination AND typed return
task = Task(
    agent,
    interactive=False,
    config=TaskConfig(done_sequences=["T[AnalysisResult], A"]),
)[AnalysisResult]

# Task terminates only on AnalysisResult
# And returns AnalysisResult | None
result: AnalysisResult | None = task.run(prompt)
```

## Multiple Tool Types

```python
from typing import Union

# Can use Union for multiple possible return types
task = Task(agent)[Union[SuccessTool, ErrorTool]]
result: SuccessTool | ErrorTool | None = task.run(prompt)

if isinstance(result, SuccessTool):
    print("Success:", result.data)
elif isinstance(result, ErrorTool):
    print("Error:", result.message)
```

## Key Points

- Subscript does NOT control which tools trigger termination
- Use `done_sequences` to control termination, subscript for typing
- Return type is `ToolType | None` (None if task fails/times out)
- Works with `run()` and `run_async()`
- Type checkers (mypy, pyright) understand the return type

## When to Use

- You want clean typed access to tool output
- Building APIs that return structured data
- Avoiding manual tool extraction from ChatDocument
- Type-safe code with IDE autocompletion
