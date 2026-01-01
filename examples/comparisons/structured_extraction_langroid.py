"""
Structured Information Extraction with Langroid.

This example demonstrates how to:
1. Define a structured output tool (PersonInfo)
2. Configure task termination with `done_if_tool=True`
3. Handle LLM forgetting to use the tool with `handle_llm_no_tool`

Run:
    python examples/comparisons/structured_extraction_langroid.py

Compare with: structured_extraction_google_adk.py
"""

from typing import List

import langroid as lr
import langroid.language_models as lm
from langroid.pydantic_v1 import Field


# =============================================================================
# 1. DEFINE THE STRUCTURED OUTPUT TOOL
# =============================================================================


class PersonInfo(lr.ToolMessage):
    """Tool for extracting structured person information from text."""

    request: str = "person_info"
    purpose: str = "Extract structured information about a person from text."

    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="The person's age in years")
    occupation: str = Field(..., description="The person's job or profession")
    location: str = Field(..., description="Where the person lives or works")

    def handle(self) -> lr.agent.tools.orchestration.ResultTool:
        """
        Handle the tool call - this runs when the LLM correctly uses the tool.
        Return a ResultTool to terminate the task with this result.
        """
        return lr.agent.tools.orchestration.ResultTool(
            person=self.dict(exclude={"request", "purpose"})
        )

    @classmethod
    def examples(cls) -> List["PersonInfo"]:
        """Few-shot examples to help the LLM understand the expected format."""
        return [
            cls(
                name="Jane Smith",
                age=32,
                occupation="Software Engineer",
                location="Seattle, WA",
            )
        ]


# =============================================================================
# 2. CREATE AGENT WITH TOOL AND TERMINATION CONFIG
# =============================================================================


def create_extraction_agent() -> lr.ChatAgent:
    """Create an agent configured for structured extraction."""

    llm_config = lm.OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4o_MINI,
        temperature=0,
    )

    agent_config = lr.ChatAgentConfig(
        name="ExtractorAgent",
        llm=llm_config,
        system_message=f"""
You are an expert at extracting structured information from text.
When given a passage about a person, you MUST use the `{PersonInfo.name()}` tool
to output the extracted information in a structured format.

IMPORTANT: Always use the tool - never respond with plain text.
""",
        # =====================================================================
        # KEY LANGROID FEATURE: handle_llm_no_tool
        # =====================================================================
        # This is triggered when the LLM responds WITHOUT using a tool.
        # Options:
        #   - A string: sent back to LLM as a reminder/nudge
        #   - "done": terminate with AgentDoneTool
        #   - "user": forward to user
        #   - A callable: custom logic based on the message
        #   - A ToolMessage: return a specific tool
        handle_llm_no_tool=f"""
You MUST use the `{PersonInfo.name()}` tool to extract person information!
Do not respond with plain text. Use the tool now.
""",
    )

    agent = lr.ChatAgent(agent_config)

    # Enable the tool - this injects tool schema and examples into the prompt
    agent.enable_message(PersonInfo)

    return agent


# =============================================================================
# 3. CREATE TASK WITH TERMINATION CONTROL
# =============================================================================


def create_extraction_task(agent: lr.ChatAgent) -> lr.Task:
    """Create a task that terminates when the tool is called."""

    task_config = lr.TaskConfig(
        # =====================================================================
        # KEY LANGROID FEATURE: done_if_tool
        # =====================================================================
        # Task terminates as soon as the LLM generates a valid tool call.
        # The tool's handle() method runs, and its return value becomes
        # the task result.
        done_if_tool=True,
    )

    # The [ResultTool] syntax creates a typed task that returns ResultTool
    task = lr.Task(
        agent,
        interactive=False,
        config=task_config,
    )[lr.agent.tools.orchestration.ResultTool]

    return task


# =============================================================================
# 4. RUN EXTRACTION
# =============================================================================


def extract_person_info(text: str) -> dict | None:
    """
    Extract structured person information from text.

    Args:
        text: A passage containing information about a person.

    Returns:
        A dictionary with name, age, occupation, location, or None if extraction fails.
    """
    agent = create_extraction_agent()
    task = create_extraction_task(agent)

    result = task.run(text)

    if result and hasattr(result, "person"):
        return result.person
    return None


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run the extraction example."""

    # Sample passages to extract from
    passages = [
        """
        Meet Dr. Sarah Chen, a 45-year-old neurosurgeon who has been practicing
        at Massachusetts General Hospital in Boston for the past 15 years.
        She specializes in minimally invasive brain surgery.
        """,
        """
        The interview featured Marcus Johnson, age 28, who works as a data
        scientist at a tech startup in Austin, Texas. He discussed his journey
        from studying physics to working in machine learning.
        """,
    ]

    print("=" * 60)
    print("LANGROID: Structured Information Extraction")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  - PersonInfo tool with Pydantic validation")
    print("  - handle_llm_no_tool: nudges LLM if it forgets the tool")
    print("  - done_if_tool=True: terminates task on successful tool call")
    print()

    for i, passage in enumerate(passages, 1):
        print(f"--- Passage {i} ---")
        print(passage.strip())
        print()

        result = extract_person_info(passage)

        if result:
            print(f"Extracted: {result}")
        else:
            print("Extraction failed!")
        print()


if __name__ == "__main__":
    main()
