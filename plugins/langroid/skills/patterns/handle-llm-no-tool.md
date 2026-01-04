# Pattern: Force Tool Usage with handle_llm_no_tool

## Problem

The LLM sometimes generates plain text instead of using the required tool.
You want to force tool-only output with an error message when this happens.

## Solution

Set `handle_llm_no_tool` in `ChatAgentConfig` to an error message string.
When the LLM responds without a tool, this message is returned to the LLM
for retry.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class DecisionTool(ToolMessage):
    """Tool for returning a structured decision."""
    request: str = "decision"
    purpose: str = "Return your decision"

    verdict: str = Field(..., description="APPROVE or REJECT")
    confidence: float = Field(..., description="Confidence 0.0-1.0")
    reasoning: str = Field(..., description="Brief explanation")


# Method 1: Simple error message
config1 = lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    system_message="Analyze and return a decision using the decision tool.",

    # Simple error message
    handle_llm_no_tool="ERROR: You must use the decision tool. Do not write plain text.",
)


# Method 2: Detailed error with tool name
config2 = lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    system_message=f"Use {DecisionTool.name()} to return your decision.",

    # Detailed error with instructions
    handle_llm_no_tool=f"""
    ERROR: You FORGOT to use the `{DecisionTool.name()}` tool!

    You MUST use this tool to return your decision.
    Do NOT write any text before the tool.
    Output ONLY the tool call, nothing else.

    Try again using the {DecisionTool.name()} tool.
    """,
)


# Method 3: Reference multiple tools
class SearchTool(ToolMessage):
    request: str = "search"
    purpose: str = "Search for information"
    query: str = Field(..., description="Search query")


config3 = lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    system_message="Search and then decide.",

    handle_llm_no_tool=f"""
    ERROR: You must use one of your tools:
    - `{SearchTool.name()}` to search for information
    - `{DecisionTool.name()}` to return your final decision

    Do not respond with plain text.
    """,
)


def run_decision_agent(prompt: str) -> DecisionTool | None:
    """Run agent that must use DecisionTool."""
    agent = lr.ChatAgent(config2)
    agent.enable_message(DecisionTool)

    task = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )[DecisionTool]

    # LLM will be reminded to use tool if it forgets
    result = task.run(prompt, turns=5)  # Allow retries
    return result
```

## Other handle_llm_no_tool Values

| Value | Behavior |
|-------|----------|
| `str` | Return message to LLM for retry (most common) |
| `"user"` | Pass message to user (for interactive tasks) |
| `"done"` | Terminate task |
| `callable` | Call function with message, use return value |

## Callable Example

```python
def custom_handler(msg: str) -> str:
    """Custom logic for no-tool responses."""
    if "I don't know" in msg:
        return "If you're unsure, use the tool with confidence=0.5"
    return "You must use the tool. Try again."


config = lr.ChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
    handle_llm_no_tool=custom_handler,
)
```

## Key Points

- Set in `ChatAgentConfig`, not `TaskConfig`
- Use f-strings with `ToolClass.name()` for dynamic tool references
- Works with task's turn limit (retries until limit or success)
- Most common use: error message string
- Combine with clear system message instructions

## When to Use

- Agent must always use structured tool output
- LLM tends to explain instead of using tools
- You need consistent structured responses
- Enforcing tool-only output in automation
