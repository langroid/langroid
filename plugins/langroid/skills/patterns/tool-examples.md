# Pattern: Tool Examples for Few-Shot Learning

## Problem

The LLM sometimes generates malformed tool calls or uses incorrect field
values. You want to improve accuracy through few-shot examples.

## Solution

Add an `examples()` classmethod to your ToolMessage that returns sample
instances showing correct usage.

## Complete Code Example

```python
from typing import List, Tuple
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


class CalculationTool(ToolMessage):
    """Tool for performing calculations."""

    request: str = "calculate"
    purpose: str = "Perform a mathematical calculation and return the result"

    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')",
    )
    result: float = Field(
        ...,
        description="The numerical result of the calculation",
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of how the result was obtained",
    )

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        """
        Return examples for few-shot learning.

        Can return:
        - Just tool instances
        - Tuples of (context_string, tool_instance)
        """
        return [
            # Simple instance
            cls(
                expression="15 + 27",
                result=42.0,
                explanation="Added 15 and 27 to get 42",
            ),
            # Instance with context (shows when to use)
            (
                "User asked: What is 8 multiplied by 7?",
                cls(
                    expression="8 * 7",
                    result=56.0,
                    explanation="Multiplied 8 by 7 using standard multiplication",
                ),
            ),
            # More complex example
            (
                "Calculate the area of a rectangle with width 5 and height 12",
                cls(
                    expression="5 * 12",
                    result=60.0,
                    explanation="Area = width × height = 5 × 12 = 60 square units",
                ),
            ),
        ]


class ClassificationTool(ToolMessage):
    """Tool for classifying content."""

    request: str = "classify"
    purpose: str = "Classify content into predefined categories"

    category: str = Field(
        ...,
        description="One of: POSITIVE, NEGATIVE, NEUTRAL",
    )
    confidence: float = Field(
        ...,
        description="Confidence score 0.0-1.0",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the classification",
    )

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        """Examples showing each category."""
        return [
            cls(
                category="POSITIVE",
                confidence=0.92,
                reasoning="Text expresses satisfaction and enthusiasm",
            ),
            cls(
                category="NEGATIVE",
                confidence=0.85,
                reasoning="Text contains complaints and frustration",
            ),
            cls(
                category="NEUTRAL",
                confidence=0.78,
                reasoning="Text is factual without emotional language",
            ),
        ]
```

## Key Points

- Return `List[ToolMessage]` or `List[Tuple[str, ToolMessage]]`
- Context strings help LLM understand when to use the tool
- Include examples covering different use cases
- Show edge cases or common patterns
- Examples are included in the prompt to the LLM

## When to Use

- LLM frequently makes mistakes with tool format
- Tool has complex field requirements
- You want more consistent LLM behavior
- Domain-specific terminology needs demonstration
