# Pattern: Sequential Multi-Agent Orchestration

## Problem

You have a workflow requiring multiple specialized agents working in
sequence - like Writer → Reviewer → Editor.

## Solution

Create separate agents and tasks, run them sequentially, passing output
from one to the next. Set agent state before running if needed.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import Field


# --- Tools ---

class DraftTool(ToolMessage):
    request: str = "submit_draft"
    purpose: str = "Submit the draft"
    content: str = Field(..., description="The draft content")


class FeedbackTool(ToolMessage):
    request: str = "submit_feedback"
    purpose: str = "Submit review feedback"
    issues: list[str] = Field(..., description="List of issues found")
    approved: bool = Field(..., description="Whether draft is approved")


class FinalTool(ToolMessage):
    request: str = "submit_final"
    purpose: str = "Submit final version"
    content: str = Field(..., description="The final content")


# --- Agent Factories ---

def create_writer() -> lr.ChatAgent:
    config = lr.ChatAgentConfig(
        name="Writer",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="Write content based on the prompt. Use submit_draft when done.",
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(DraftTool)
    return agent


def create_reviewer() -> lr.ChatAgent:
    config = lr.ChatAgentConfig(
        name="Reviewer",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="""
        Review the draft for issues. Use submit_feedback with:
        - issues: list of problems found
        - approved: true if ready, false if needs revision
        """,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(FeedbackTool)
    return agent


def create_editor() -> lr.ChatAgent:
    config = lr.ChatAgentConfig(
        name="Editor",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="Edit the draft to fix the issues. Use submit_final when done.",
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(FinalTool)
    return agent


# --- Orchestration ---

def run_workflow(prompt: str, max_rounds: int = 3) -> str:
    """Run Writer → Reviewer → Editor workflow."""

    # Phase 1: Initial Draft
    writer = create_writer()
    writer_task = Task(
        writer,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )[DraftTool]

    draft_result = writer_task.run(prompt)
    if not draft_result:
        raise RuntimeError("Writer failed to produce draft")

    current_draft = draft_result.content

    # Phase 2 & 3: Review and Edit Loop
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Review Round {round_num} ---")

        # Review
        reviewer = create_reviewer()
        reviewer_task = Task(
            reviewer,
            interactive=False,
            config=TaskConfig(done_if_tool=True),
        )[FeedbackTool]

        feedback = reviewer_task.run(f"Review this draft:\n\n{current_draft}")
        if not feedback:
            raise RuntimeError("Reviewer failed")

        if feedback.approved:
            print("Draft approved!")
            return current_draft

        # Edit
        editor = create_editor()
        editor_task = Task(
            editor,
            interactive=False,
            config=TaskConfig(done_if_tool=True),
        )[FinalTool]

        edit_prompt = f"""
        Original draft:
        {current_draft}

        Issues to fix:
        {chr(10).join(f'- {issue}' for issue in feedback.issues)}

        Fix these issues and submit the final version.
        """

        edit_result = editor_task.run(edit_prompt)
        if not edit_result:
            raise RuntimeError("Editor failed")

        current_draft = edit_result.content

    return current_draft


# Run the workflow
final = run_workflow("Write a short blog post about Python async/await.")
print(final)
```

## Key Points

- Create fresh agents for each phase (clean state)
- Use `TaskConfig(done_if_tool=True)` for simple termination
- Pass output from one task as input to the next
- Use subscript notation for typed returns
- Handle failures at each step
- Limit iterations to prevent infinite loops

## Passing State to Agents

```python
# If agent needs state, set it before running
editor = EditorAgent(config)
editor.enable_message(EditTool)
editor.original_text = draft_content  # Set state
editor.required_markers = ["[PLACEHOLDER]"]

result = editor_task.run(prompt)
```

## When to Use

- Workflows with distinct phases (draft, review, edit)
- Each phase has different expertise/tools
- Output of one phase is input to the next
- Need visibility/control between phases
