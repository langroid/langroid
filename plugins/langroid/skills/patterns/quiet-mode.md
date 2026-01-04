# Pattern: Quiet Mode for Clean Output

## Problem

Langroid agents produce verbose output (streaming, tool JSON, intermediate
messages). You want clean CLI output with only your custom progress messages.

## Solution

Use the `quiet_mode()` context manager to suppress Langroid output during
task execution, then print your own messages outside the context.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field
from langroid.utils.output import quiet_mode
from rich.console import Console


class AnalysisTool(ToolMessage):
    request: str = "analysis"
    purpose: str = "Return analysis"
    result: str = Field(..., description="Analysis result")


def run_analysis(data: str) -> str | None:
    """Run analysis with clean output."""
    console = Console()

    agent = lr.ChatAgent(lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="Analyze the data and return your analysis.",
    ))
    agent.enable_message(AnalysisTool)

    task = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )[AnalysisTool]

    # Show progress BEFORE quiet mode
    console.print("[blue]Starting analysis...[/blue]")

    # Suppress Langroid output during task execution
    with quiet_mode():
        result = task.run(f"Analyze this: {data}")

    # Show result AFTER quiet mode
    if result:
        console.print("[green]Analysis complete![/green]")
        return result.result
    else:
        console.print("[red]Analysis failed[/red]")
        return None


def run_multi_step_workflow(items: list[str]) -> list[str]:
    """Multi-step workflow with progress updates."""
    console = Console()
    results = []

    agent = lr.ChatAgent(lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="Process the item.",
    ))
    agent.enable_message(AnalysisTool)

    for i, item in enumerate(items, 1):
        # Progress OUTSIDE quiet mode
        console.print(f"[blue]Processing item {i}/{len(items)}...[/blue]")

        task = Task(
            agent,
            interactive=False,
            config=TaskConfig(done_if_tool=True),
        )[AnalysisTool]

        # Suppress during execution
        with quiet_mode():
            result = task.run(item)

        if result:
            results.append(result.result)
            console.print(f"[green]  Done[/green]")
        else:
            console.print(f"[red]  Failed[/red]")

        # Reset agent state for next item
        agent.init_state()

    return results
```

## What quiet_mode() Suppresses

- LLM streaming output
- Tool JSON/function call details
- Intermediate agent messages
- Debug/info logging from Langroid

## What It Does NOT Suppress

- Your `print()` or `console.print()` statements
- Errors and exceptions
- Output from code outside the context manager

## Alternative: Per-Agent Streaming Control

```python
# Disable streaming at LLM config level
config = lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model="gpt-4o",
        stream=False,  # Disable streaming
    ),
)
```

## Async Usage

```python
async def run_async_with_quiet():
    with quiet_mode():
        result = await task.run_async(prompt)
    return result
```

## Key Points

- Import: `from langroid.utils.output import quiet_mode`
- Use `with quiet_mode():` around `task.run()` calls
- Print progress messages OUTSIDE the context manager
- Works with both `run()` and `run_async()`
- Thread-safe for concurrent task execution

## When to Use

- CLI applications needing clean output
- Multi-step workflows with progress reporting
- Batch processing with status updates
- Production deployments where verbose output is unwanted
