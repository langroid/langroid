# Pattern: Agent with Custom State

## Problem

Your tool handler needs access to state that persists across tool calls -
connections, counters, accumulated data, or context from the original request.

## Solution

Create a `ChatAgent` subclass with custom `__init__` that stores state as
instance attributes. Handler methods can then access `self.state_var`.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.pydantic_v1 import Field


class EditTool(ToolMessage):
    """Tool for submitting an edit."""
    request: str = "submit_edit"
    purpose: str = "Submit the edited text"
    edited_text: str = Field(..., description="The edited version")


class EditorAgent(lr.ChatAgent):
    """Agent that validates edits preserve required content."""

    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        # Custom state - will be set before running
        self.original_text: str = ""
        self.required_markers: list[str] = []

    def init_state(self) -> None:
        """Reset state between tasks. Called by Langroid."""
        super().init_state()
        self.original_text = ""
        self.required_markers = []

    def submit_edit(self, msg: EditTool) -> str | AgentDoneTool:
        """
        Handler for EditTool. Validates that required markers are preserved.

        Returns:
            str: Error message (LLM retries)
            AgentDoneTool: Success (task terminates)
        """
        # Check each required marker
        for marker in self.required_markers:
            if marker in self.original_text and marker not in msg.edited_text:
                return (
                    f"ERROR: You removed the marker '{marker}'. "
                    f"This marker MUST be preserved. Try again."
                )

        # All markers preserved - success
        return AgentDoneTool(tools=[msg])


def run_editor(original: str, markers: list[str], instructions: str) -> str | None:
    """Run the editor agent with state."""
    config = lr.ChatAgentConfig(
        name="EditorAgent",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message=f"""
        Edit the text according to the instructions.
        IMPORTANT: You must preserve these markers exactly: {markers}

        Use the submit_edit tool when done.
        """,
    )

    agent = EditorAgent(config)
    agent.enable_message(EditTool)

    # Set state BEFORE running
    agent.original_text = original
    agent.required_markers = markers

    task = lr.Task(
        agent,
        interactive=False,
        config=lr.TaskConfig(done_sequences=["T[EditTool], A"]),
    )[EditTool]

    result = task.run(f"Original text:\n{original}\n\nInstructions: {instructions}")
    return result.edited_text if result else None
```

## Key Points

- Override `__init__` but always call `super().__init__(config)`
- Override `init_state()` to reset state between task runs
- Set state attributes BEFORE calling `task.run()`
- Handler method name must match tool's `request` field
- Return `str` for retry, `AgentDoneTool` for success

## When to Use

- Tool handler needs validation against input context
- Tracking state across multiple tool calls
- Handler needs access to external resources (DB connections, APIs)
- Complex multi-step workflows with intermediate state
