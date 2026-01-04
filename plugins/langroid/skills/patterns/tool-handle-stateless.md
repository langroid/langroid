# Pattern: Stateless Tool Handler (in ToolMessage)

## Problem

You need to validate or transform tool output without needing agent state.
The logic is simple and self-contained.

## Solution

Define a `handle()` method directly inside the ToolMessage class. Return
a string for the LLM to see, or a special tool to control task flow.

## Complete Code Example

```python
import re
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.pydantic_v1 import Field


class EmailTool(ToolMessage):
    """Tool for submitting an email address."""

    request: str = "submit_email"
    purpose: str = "Submit a validated email address"

    email: str = Field(..., description="The email address")

    def handle(self) -> str | AgentDoneTool:
        """
        Validate email format.

        Returns:
            str: Error message (LLM retries)
            AgentDoneTool: Success with validated email
        """
        # Simple email validation
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(pattern, self.email):
            return (
                f"ERROR: '{self.email}' is not a valid email format. "
                f"Please provide a valid email like 'user@example.com'"
            )

        # Valid - terminate with this tool
        return AgentDoneTool(tools=[self])


class CalculatorTool(ToolMessage):
    """Tool for safe arithmetic calculations."""

    request: str = "calculate"
    purpose: str = "Perform safe arithmetic calculation"

    expression: str = Field(..., description="Math expression like '2 + 3 * 4'")

    def handle(self) -> str:
        """
        Safely evaluate the expression.

        Returns:
            str: Result or error message (continues conversation)
        """
        # Only allow safe characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', self.expression):
            return f"ERROR: Expression contains invalid characters: {self.expression}"

        try:
            result = eval(self.expression)
            return f"Result: {self.expression} = {result}"
        except Exception as e:
            return f"ERROR: Could not evaluate '{self.expression}': {e}"


class ProofreadTool(ToolMessage):
    """Tool for submitting proofread text."""

    request: str = "submit_proofread"
    purpose: str = "Submit proofread text"

    text: str = Field(..., description="The proofread text")
    changes_made: int = Field(..., description="Number of changes made")

    def handle(self) -> str | AgentDoneTool:
        """Validate the proofread submission."""
        # Check minimum length
        if len(self.text.strip()) < 10:
            return "ERROR: Text is too short. Please provide the full proofread text."

        # Check changes were actually made
        if self.changes_made < 0:
            return "ERROR: changes_made cannot be negative."

        # Success
        return AgentDoneTool(tools=[self])
```

## Return Value Behavior

| Return Type | Behavior |
|-------------|----------|
| `str` | Message goes to LLM, conversation continues |
| `AgentDoneTool(tools=[self])` | Task terminates with this tool as result |
| `DoneTool(content=...)` | Task terminates with content |
| Another `ToolMessage` | That tool gets processed next |

## Key Points

- `handle()` method takes no parameters (uses `self.field`)
- Return `str` for errors/feedback that LLM should see
- Return `AgentDoneTool(tools=[self])` for successful termination
- Handler runs automatically when LLM emits the tool
- No access to agent state (use agent handler for that)

## When to Use

- Simple validation that doesn't need agent context
- Pure transformations of tool data
- Self-contained logic without side effects
- Quick checks before accepting tool output
