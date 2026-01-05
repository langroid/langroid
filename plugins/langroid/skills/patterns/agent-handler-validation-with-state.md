# Pattern: Validate Tool Output Against Agent State

## Problem

You have an agent that produces tool output, but you need to validate that output
against the input context before accepting it. For example:
- Ensuring placeholders like `{{differentiation}}` are preserved in edited text
- Verifying required fields aren't removed
- Checking that certain patterns from the input appear in the output

If validation fails, you want the LLM to automatically retry.

## Solution

1. Create a **custom agent class** that stores input context as state
2. Define a **handler method** on the agent (name matches tool's `request` field)
3. In the handler, **validate** tool output against stored state
4. Return **error string** for retry, or **AgentDoneTool** for success
5. Use `done_sequences=["T[ToolName], A"]` so handler runs before task terminates
   (use `["T, A"]` only if agent has a single unambiguous tool)

## Complete Code Example

```python
import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from pydantic import Field


# Reserved content that must be preserved
RESERVED_PLACEHOLDERS = ["{{differentiation}}", "{{company_info}}"]


class LineReplacementTool(ToolMessage):
    """Tool for LLM to output replacement text."""
    request: str = "emit_line_replacement"
    purpose: str = "Output the replacement text for the specified lines"

    replacement_text: str = Field(..., description="The new text")
    explanation: str = Field(..., description="Brief explanation of the edit")


class LineEditorAgent(ChatAgent):
    """Editor agent that validates placeholder preservation."""

    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.current_text: str = ""  # Set before task.run()

    def init_state(self):
        """Reset state between tasks."""
        super().init_state()
        self.current_text = ""

    def emit_line_replacement(self, msg: LineReplacementTool) -> str | AgentDoneTool:
        """
        Handler for LineReplacementTool. Validates placeholder preservation.

        Name matches the tool's `request` field exactly.
        """
        # Check if any reserved placeholder in original is missing from replacement
        for placeholder in RESERVED_PLACEHOLDERS:
            if placeholder in self.current_text:
                if placeholder not in msg.replacement_text:
                    # Return error string - LLM sees this and can retry
                    return (
                        f"ERROR: You removed the placeholder {placeholder}. "
                        f"This placeholder MUST be preserved exactly as-is. "
                        f"Please output the replacement again, keeping {placeholder} intact."
                    )

        # Validation passed - terminate task successfully
        # Return AgentDoneTool with the validated tool in the tools list
        return AgentDoneTool(tools=[msg])


def create_editor_agent(model: str) -> LineEditorAgent:
    """Create the editor agent with validation handler."""
    config = ChatAgentConfig(
        name="LineEditor",
        llm=lr.language_models.OpenAIGPTConfig(chat_model=model),
        system_message="""You are a precise technical editor.
You will receive text to edit along with instructions.
Output the replacement using the emit_line_replacement tool.
IMPORTANT: Preserve any {{...}} placeholders exactly as they appear.""",
    )
    agent = LineEditorAgent(config)
    agent.enable_message(LineReplacementTool)
    return agent


def apply_edit(current_text: str, instruction: str, model: str) -> LineReplacementTool | None:
    """Apply an edit with placeholder validation."""
    agent = create_editor_agent(model)

    # Store current text in agent state for handler to access
    agent.current_text = current_text

    # Use done_sequences so handler runs before task terminates
    # "T[ToolName], A" = Specific tool emitted, then Agent handles it
    # Use "T, A" only if agent has a single unambiguous tool
    task = lr.Task(
        agent,
        interactive=False,
        config=lr.TaskConfig(done_sequences=["T[LineReplacementTool], A"]),
    )[LineReplacementTool]

    prompt = f"""Edit this text:

{current_text}

Instruction: {instruction}

Use emit_line_replacement tool with your replacement."""

    # If handler returns error string, LLM retries automatically
    # If handler returns DoneTool, task terminates and we get the tool
    result: LineReplacementTool | None = task.run(prompt, turns=5)
    return result
```

## Key Points

1. **Handler method name = tool's `request` field**: If `request = "emit_line_replacement"`,
   define `def emit_line_replacement(self, msg)`

2. **Store context before task.run()**: Set `agent.current_text = ...` so handler can access it

3. **Return types control flow**:
   - `str` (error message) → Langroid sends to LLM, triggers retry
   - `AgentDoneTool(tools=[msg])` → Task terminates successfully with the tool
   - Note: Use `AgentDoneTool` (has `tools` field), NOT `DoneTool` (no `tools` field)

4. **done_sequences=["T[ToolName], A"]**: Ensures handler runs. Without this, task
   might exit immediately when tool is emitted, skipping validation. Use `["T, A"]`
   only when agent has a single unambiguous tool.

5. **init_state()**: Override to reset state between uses if agent is reused

## When to Use This Pattern

- LLM must preserve certain content (placeholders, markers, required fields)
- You need to validate output against input context
- Validation failure should trigger automatic retry
- Simple prompt instructions aren't reliable enough (small LLMs ignore them)
