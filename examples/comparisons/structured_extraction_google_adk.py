"""
Structured Information Extraction with Google ADK.

This example demonstrates the equivalent task in Google's Agent Development Kit,
showing the IDIOMATIC way to handle:
1. Tool definition
2. Task termination (implicit in ADK)
3. Handling LLM forgetting to use tools (requires callback + retry loop)

Run:
    python examples/comparisons/structured_extraction_google_adk.py

Compare with: structured_extraction_langroid.py

Requirements:
    pip install google-adk
    export GOOGLE_API_KEY="your-key"
"""

import asyncio
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# =============================================================================
# 1. DEFINE THE TOOL
# =============================================================================
# In Google ADK, tools are plain Python functions with type hints.
# The framework infers the schema from the function signature and docstring.


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
    return {
        "name": name,
        "age": age,
        "occupation": occupation,
        "location": location,
    }


# =============================================================================
# 2. CALLBACK TO DETECT MISSING TOOL CALLS
# =============================================================================
# This is the IDIOMATIC Google ADK approach to handle "LLM forgot to use tool".
#
# What Langroid does in 1 line:
#     handle_llm_no_tool = "You FORGOT to use the tool!"
#
# Google ADK requires this callback class + application-level retry logic.


def has_function_call(llm_response: Any) -> bool:
    """
    Check if an LLM response contains a function call.

    Google ADK responses can have different structures depending on the
    model provider, so we check multiple patterns defensively.
    """
    try:
        # Pattern 1: Check via candidates (Gemini native format)
        if hasattr(llm_response, "candidates"):
            for candidate in llm_response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            return True

        # Pattern 2: Direct content access (simplified format)
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    return True
    except Exception:
        pass
    return False


class ToolEnforcementCallback:
    """
    After-model callback that detects when LLM forgets to use a required tool.

    This implements what Langroid's `handle_llm_no_tool` does automatically.
    Uses callback_context.state for retry tracking (the idiomatic ADK pattern).

    Usage:
        callback = ToolEnforcementCallback(
            nudge_message="You MUST use the tool!",
            max_retries=3,
        )
        agent = LlmAgent(after_model_callback=callback, ...)
    """

    def __init__(
        self,
        nudge_message: str = "You MUST use one of your available tools!",
        max_retries: int = 3,
    ):
        self.nudge_message = nudge_message
        self.max_retries = max_retries

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """
        Inspect LLM response and flag if tool was not used.

        Note: Google ADK callbacks cannot directly trigger a retry.
        We set state flags that the application-level code must check.

        Returns:
            None - always let the response through (we handle retry externally)
        """
        # Access state via callback_context (idiomatic ADK pattern)
        state = callback_context.state

        # Initialize retry tracking if needed
        if "_tool_retry_count" not in state:
            state["_tool_retry_count"] = 0

        # Check if tool was called
        if has_function_call(llm_response):
            # Success! Reset retry counter
            state["_tool_retry_count"] = 0
            state["_tool_was_used"] = True
            return None

        # No tool call detected
        retry_count = state["_tool_retry_count"]
        state["_tool_was_used"] = False

        if retry_count >= self.max_retries:
            print(f"WARNING: LLM failed to use tool after {self.max_retries} retries")
            state["_max_retries_exceeded"] = True
            return None

        # Flag for retry (application code must handle this)
        state["_tool_retry_count"] = retry_count + 1
        state["_needs_tool_retry"] = True
        state["_nudge_message"] = self.nudge_message

        print(
            f"  [Callback] No tool call detected "
            f"(attempt {retry_count + 1}/{self.max_retries})"
        )

        return None  # Let response through - retry handled at app level


# =============================================================================
# 3. CREATE THE AGENT
# =============================================================================


def create_extraction_agent() -> LlmAgent:
    """
    Create an agent configured for structured extraction.

    Comparison of setup complexity:

    LANGROID (built-in tool enforcement):
        config = lr.ChatAgentConfig(
            handle_llm_no_tool="You MUST use the tool!",
        )
        # That's it - 1 line handles detection + retry

    GOOGLE ADK (manual tool enforcement):
        callback = ToolEnforcementCallback(...)  # ~40 line class
        agent = LlmAgent(after_model_callback=callback, ...)
        # Plus ~50 lines of application-level retry logic below
    """
    callback = ToolEnforcementCallback(
        nudge_message=(
            "You MUST use the `extract_person_info` tool to extract information. "
            "Do NOT respond with plain text. Call the tool NOW."
        ),
        max_retries=3,
    )

    return LlmAgent(
        name="extractor_agent",
        model="gemini-2.0-flash",
        instruction="""
You are an expert at extracting structured information from text.
When given a passage about a person, you MUST use the `extract_person_info` tool
to output the extracted information in a structured format.

CRITICAL: Always use the tool. Never respond with plain text.
""",
        tools=[extract_person_info],
        after_model_callback=callback,
    )


# =============================================================================
# 4. APPLICATION-LEVEL RETRY LOGIC
# =============================================================================
# Google ADK callbacks CANNOT trigger retries directly.
# The application must implement retry logic by:
# 1. Running the agent
# 2. Checking if tool was used (via state flags set by callback)
# 3. If not, sending a new message with the nudge prepended
#
# This is what Langroid handles automatically with handle_llm_no_tool.


async def extract_person_info_async(
    text: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Extract structured person information with retry on missing tool calls.

    This function implements the retry loop that Langroid provides automatically.

    Args:
        text: A passage containing information about a person.
        max_retries: Maximum retry attempts if LLM forgets to use tool.

    Returns:
        Extracted person info dict, or None if extraction fails.
    """
    agent = create_extraction_agent()
    session_service = InMemorySessionService()

    for attempt in range(max_retries + 1):
        # Create a fresh session for each attempt
        # (In production, you might reuse sessions with conversation history)
        session = await session_service.create_session(
            app_name="extraction_demo",
            user_id="demo_user",
        )

        runner = Runner(
            agent=agent,
            app_name="extraction_demo",
            session_service=session_service,
        )

        # Build the message - prepend nudge on retry attempts
        if attempt == 0:
            message_text = text
        else:
            message_text = f"""
IMPORTANT REMINDER: You MUST use the `extract_person_info` tool!
Do NOT respond with plain text. Use the tool to extract the information.

Here is the text to extract from:

{text}
"""
            print(f"  [App] Retry {attempt}/{max_retries}: Sending nudge...")

        # Run the agent and collect results
        result = None
        tool_was_used = False

        async for event in runner.run_async(
            session_id=session.id,
            user_id="demo_user",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=message_text)],
            ),
        ):
            # Check events for tool usage
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    # Tool was called and returned response
                    if hasattr(part, "function_response") and part.function_response:
                        result = part.function_response.response
                        tool_was_used = True
                    # Tool is being called (function_call event)
                    elif hasattr(part, "function_call") and part.function_call:
                        tool_was_used = True

        # Success - tool was used and we got a result
        if tool_was_used and result is not None:
            return result

        # Check if we should retry
        if attempt >= max_retries:
            print("  [App] Max retries exceeded - extraction failed")
            break

    return None


def extract_person(text: str) -> dict | None:
    """Synchronous wrapper for the async extraction function."""
    return asyncio.run(extract_person_info_async(text))


# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================


def main():
    """Run the extraction example and show the comparison."""

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

    print("=" * 70)
    print("GOOGLE ADK: Structured Information Extraction")
    print("=" * 70)
    print()
    print("This example shows the IDIOMATIC way to handle 'LLM forgot tool' in ADK:")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ LANGROID (built-in):                                                │")
    print("│   handle_llm_no_tool = 'You MUST use the tool!'                     │")
    print("│   # Framework handles detection + retry automatically               │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ GOOGLE ADK (manual):                                                │")
    print("│   1. ToolEnforcementCallback class (~40 lines)                      │")
    print("│   2. has_function_call() helper (~20 lines)                         │")
    print("│   3. Application-level retry loop (~50 lines)                       │")
    print("│   # Developer must implement all retry logic                        │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    for i, passage in enumerate(passages, 1):
        print(f"--- Passage {i} ---")
        print(passage.strip())
        print()

        result = extract_person(passage)

        if result:
            print(f"✓ Extracted: {result}")
        else:
            print("✗ Extraction failed!")
        print()


if __name__ == "__main__":
    main()
