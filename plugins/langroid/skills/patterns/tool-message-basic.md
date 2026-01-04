# Pattern: Basic ToolMessage Definition

## Problem

You need the LLM to generate structured output with specific fields,
validated by Pydantic.

## Solution

Create a `ToolMessage` subclass with `request`, `purpose`, and typed fields
using `Field()` for descriptions.

## Complete Code Example

```python
from typing import Optional, List
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class ExtractedInfo(ToolMessage):
    """Tool for extracting structured information from text."""

    # Required: identifies the tool
    request: str = "extract_info"

    # Required: tells LLM when to use this tool
    purpose: str = "Extract structured information from the provided text"

    # Required fields (... means required)
    title: str = Field(
        ...,
        description="Main title or subject extracted from the text",
    )
    summary: str = Field(
        ...,
        description="2-3 sentence summary of the key points",
    )

    # Optional fields with defaults
    confidence: float = Field(
        0.8,
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    categories: List[str] = Field(
        default_factory=list,
        description="List of relevant categories or tags",
    )
    source_url: Optional[str] = Field(
        None,
        description="Source URL if mentioned in the text",
    )


# Usage in agent
import langroid as lr

agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="Extract information from text using the extract_info tool.",
    )
)
agent.enable_message(ExtractedInfo)

task = lr.Task(agent, interactive=False)
result = task.run("Analyze this article: [article text here]")
```

## Field Options

| Option | Description |
|--------|-------------|
| `...` | Required field (no default) |
| `default` | Default value if not provided |
| `default_factory` | Factory for mutable defaults (list, dict) |
| `description` | Description shown to LLM |
| `ge`, `le` | Numeric constraints (greater/less than or equal) |

## Key Points

- `request` field defines the tool's identifier (used in JSON)
- `purpose` field tells LLM when/why to use this tool
- Use `Field(...)` for required fields with descriptions
- Use `Optional[T]` with `None` default for optional fields
- Import from `langroid.pydantic_v1` for Pydantic compatibility

## When to Use

- Any structured output from LLM (forms, extraction, decisions)
- Data validation is needed
- Multiple related fields should be returned together
