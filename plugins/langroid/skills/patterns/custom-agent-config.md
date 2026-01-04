# Pattern: Custom Config Subclass

## Problem

You have multiple agents that share similar configuration (system message
structure, LLM settings, tool references). You want to avoid repeating the
same config code.

## Solution

Create a custom `ChatAgentConfig` subclass with preset fields and f-string
interpolation for tool names.

## Complete Code Example

```python
import langroid as lr
from langroid.language_models import OpenAIGPTConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class AnalysisTool(ToolMessage):
    """Tool for returning analysis results."""
    request: str = "analysis_result"
    purpose: str = "Return structured analysis"
    summary: str = Field(..., description="Summary of findings")
    confidence: float = Field(..., description="Confidence 0.0-1.0")


class AnalysisAgentConfig(lr.ChatAgentConfig):
    """Reusable config for analysis agents."""

    name: str = "AnalysisAgent"

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        chat_model="gpt-4o",
        temperature=0.3,  # Lower for more consistent output
        timeout=90,
    )

    # Reference tool name dynamically
    system_message: str = f"""
    You are an expert analyst. Examine the provided data carefully.

    When you have completed your analysis, use the `{AnalysisTool.name()}`
    tool to return your findings in a structured format.

    Be thorough and provide clear reasoning.
    """

    # Force tool usage
    handle_llm_no_tool: str = f"""
    ERROR: You must use the `{AnalysisTool.name()}` tool to return results.
    Do not write plain text. Output ONLY the tool call.
    """


def create_analysis_agent(custom_instructions: str = "") -> lr.ChatAgent:
    """Factory function to create configured analysis agent."""
    config = AnalysisAgentConfig()

    # Optionally append custom instructions
    if custom_instructions:
        config.system_message += f"\n\nAdditional instructions:\n{custom_instructions}"

    agent = lr.ChatAgent(config)
    agent.enable_message(AnalysisTool)
    return agent
```

## Key Points

- Use `ToolClass.name()` to reference tool names in f-strings
- Set defaults for all fields to make config reusable
- Use factory functions to customize config at creation time
- Subclass enables IDE autocompletion and type checking

## When to Use

- Multiple agents with similar structure but different details
- Team projects where config consistency matters
- Building agent libraries or frameworks
