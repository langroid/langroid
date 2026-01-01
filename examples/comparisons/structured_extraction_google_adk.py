"""
Structured Information Extraction with Google ADK.

This example demonstrates the equivalent task in Google's Agent Development Kit,
highlighting the differences in:
1. Tool definition
2. Task termination (implicit vs explicit)
3. Handling LLM forgetting to use tools (no built-in mechanism)

Run:
    python examples/comparisons/structured_extraction_google_adk.py

Compare with: structured_extraction_langroid.py

Requirements:
    pip install google-adk
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# =============================================================================
# 1. DEFINE THE STRUCTURED OUTPUT TOOL
# =============================================================================
# In Google ADK, tools are just Python functions with type hints.
# The framework infers the schema from the function signature.


@dataclass
class PersonInfo:
    """Structured person information."""

    name: str
    age: int
    occupation: str
    location: str


def extract_person_info(
    name: str,
    age: int,
    occupation: str,
    location: str,
) -> dict:
    """
    Extract and return structured person information.

    Args:
        name: The person's full name
        age: The person's age in years
        occupation: The person's job or profession
        location: Where the person lives or works

    Returns:
        A dictionary containing the extracted person information.
    """
    # In Google ADK, the tool function just returns data.
    # There's no built-in way to signal "task complete" from here.
    return {
        "name": name,
        "age": age,
        "occupation": occupation,
        "location": location,
    }


# =============================================================================
# 2. CREATE AGENT
# =============================================================================
# Note: Google ADK has NO equivalent to Langroid's `handle_llm_no_tool`.
# If the LLM forgets to call the tool, the task simply completes with
# the LLM's text response - there's no automatic retry or nudge.


def create_extraction_agent() -> LlmAgent:
    """Create an agent configured for structured extraction."""

    agent = LlmAgent(
        name="extractor_agent",
        model="gemini-2.0-flash",
        instruction="""
You are an expert at extracting structured information from text.
When given a passage about a person, you MUST use the `extract_person_info` tool
to output the extracted information in a structured format.

IMPORTANT: Always use the tool - never respond with plain text.
""",
        tools=[extract_person_info],
        # =====================================================================
        # GOOGLE ADK: No built-in handle_llm_no_tool equivalent!
        # =====================================================================
        # If you need to handle the case where the LLM forgets to use a tool,
        # you must implement a custom callback. See below for a workaround.
        #
        # Available callbacks (but none handle "forgot tool" automatically):
        #   - before_model_callback: runs before LLM call
        #   - after_model_callback: runs after LLM response (could check here)
        #   - on_model_error_callback: only for errors, not missing tools
    )

    return agent


# =============================================================================
# 3. WORKAROUND: Custom callback to detect missing tool calls
# =============================================================================
# This is the boilerplate you'd need to write in Google ADK to achieve
# what Langroid does with a single `handle_llm_no_tool` config option.


def create_agent_with_tool_nudge() -> LlmAgent:
    """
    Create an agent with a custom callback to nudge the LLM if it forgets tools.

    This demonstrates the manual workaround needed in Google ADK.
    """

    def check_for_tool_usage(
        callback_context: Any,
        llm_response: Any,
    ) -> Any:
        """
        After-model callback to check if the LLM used a tool.

        NOTE: This is a simplified example. In practice, you'd need to:
        1. Check if the response contains function calls
        2. If not, return a modified response that prompts retry
        3. Handle the retry loop yourself
        """
        # Check if response has function calls
        if hasattr(llm_response, "candidates"):
            for candidate in llm_response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            # Tool was called - let it proceed
                            return None

        # No tool call detected - in a real implementation, you'd:
        # 1. Log a warning
        # 2. Potentially inject a retry prompt
        # 3. Return a modified response
        #
        # But Google ADK doesn't make this easy - you'd need to construct
        # a new LlmResponse with a nudge message, which is non-trivial.
        print("WARNING: LLM did not use a tool! (No automatic retry in Google ADK)")
        return None

    agent = LlmAgent(
        name="extractor_agent_with_nudge",
        model="gemini-2.0-flash",
        instruction="""
You are an expert at extracting structured information from text.
When given a passage about a person, you MUST use the `extract_person_info` tool
to output the extracted information in a structured format.

IMPORTANT: Always use the tool - never respond with plain text.
""",
        tools=[extract_person_info],
        after_model_callback=check_for_tool_usage,
    )

    return agent


# =============================================================================
# 4. RUN EXTRACTION
# =============================================================================
# In Google ADK, you use a Runner to execute agents.
# Task termination is IMPLICIT: the task ends when the LLM stops calling tools.


async def extract_person_info_async(text: str) -> dict | None:
    """
    Extract structured person information from text using Google ADK.

    Args:
        text: A passage containing information about a person.

    Returns:
        A dictionary with extracted info, or None if extraction fails.

    Note:
        Google ADK task termination is implicit:
        - Task continues while LLM makes tool calls
        - Task ends when LLM responds with text (no tool calls)

        This means if the LLM "forgets" to use a tool and just responds
        with text, the task completes with that text response - NOT with
        structured data. There's no automatic retry.
    """
    agent = create_extraction_agent()

    # Google ADK requires a session service
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="extraction_demo",
        user_id="demo_user",
    )

    # Create runner
    runner = Runner(
        agent=agent,
        app_name="extraction_demo",
        session_service=session_service,
    )

    # Run the agent
    # =========================================================================
    # KEY DIFFERENCE: No explicit termination control!
    # =========================================================================
    # In Langroid: done_if_tool=True explicitly terminates on tool call
    # In Google ADK: Task runs until LLM stops calling tools (implicit)
    #
    # The loop in base_llm_flow.py checks is_final_response() which returns
    # True when there are no function calls in the response.
    result = None
    async for event in runner.run_async(
        session_id=session.id,
        user_id="demo_user",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=text)],
        ),
    ):
        # Events stream as the agent runs
        # We need to capture the tool response if it exists
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    # Tool was called and returned a response
                    result = part.function_response.response
                elif hasattr(part, "text") and part.text:
                    # LLM responded with text - might be final answer or forgot tool
                    if result is None:
                        # No tool was called - LLM might have forgotten!
                        print(f"LLM text response (no tool used): {part.text[:100]}...")

    return result


def extract_person(text: str) -> dict | None:
    """Sync wrapper for the async extraction function."""
    return asyncio.run(extract_person_info_async(text))


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run the extraction example."""

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
    print("GOOGLE ADK: Structured Information Extraction")
    print("=" * 60)
    print()
    print("Key Differences from Langroid:")
    print("  - Tools are plain Python functions (vs ToolMessage classes)")
    print("  - NO handle_llm_no_tool: must write custom callbacks")
    print("  - Implicit termination: ends when LLM stops calling tools")
    print("  - If LLM forgets tool, task completes with text (no retry!)")
    print()

    for i, passage in enumerate(passages, 1):
        print(f"--- Passage {i} ---")
        print(passage.strip())
        print()

        result = extract_person(passage)

        if result:
            print(f"Extracted: {result}")
        else:
            print("Extraction failed or LLM didn't use the tool!")
        print()


if __name__ == "__main__":
    main()
