"""
Structured Information Extraction with Google ADK.

This example demonstrates the equivalent task in Google's Agent Development Kit,
highlighting the differences in:
1. Tool definition
2. Task termination (implicit vs explicit)
3. Handling LLM forgetting to use tools (requires custom callback + retry logic)

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
    return {
        "name": name,
        "age": age,
        "occupation": occupation,
        "location": location,
    }


# =============================================================================
# 2. THE BOILERPLATE: Custom callback to nudge LLM on missing tool calls
# =============================================================================
# This is what you need to write in Google ADK to achieve what Langroid does
# with a single config option: handle_llm_no_tool="You forgot to use the tool!"
#
# In Langroid:
#     handle_llm_no_tool = "You FORGOT to use the tool! Try again."
#
# In Google ADK: ~50 lines of callback + state management code below.


class ToolNudgeCallback:
    """
    Callback class to detect when LLM forgets to use a tool and inject a nudge.

    This implements what Langroid's `handle_llm_no_tool` does automatically.

    Usage:
        callback = ToolNudgeCallback(
            nudge_message="You MUST use the extract_person_info tool!",
            max_retries=3,
        )
        agent = LlmAgent(
            after_model_callback=callback,
            ...
        )
    """

    def __init__(
        self,
        nudge_message: str = "You MUST use one of your tools! Do not respond with plain text.",
        max_retries: int = 3,
    ):
        self.nudge_message = nudge_message
        self.max_retries = max_retries
        # Track retries per session (in production, use proper state management)
        self._retry_counts: dict[str, int] = {}

    def _has_function_call(self, llm_response: Any) -> bool:
        """Check if the LLM response contains a function call."""
        try:
            if hasattr(llm_response, "candidates"):
                for candidate in llm_response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                return True
            # Also check direct content access pattern
            if hasattr(llm_response, "content") and llm_response.content:
                for part in llm_response.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        return True
        except Exception:
            pass
        return False

    def _get_session_id(self, callback_context: Any) -> str:
        """Extract session ID from callback context for retry tracking."""
        try:
            if hasattr(callback_context, "invocation_context"):
                ctx = callback_context.invocation_context
                if hasattr(ctx, "session") and ctx.session:
                    return ctx.session.id
        except Exception:
            pass
        return "default"

    def __call__(
        self,
        callback_context: Any,
        llm_response: Any,
    ) -> Any:
        """
        After-model callback that nudges the LLM if it didn't use a tool.

        Returns:
            None to proceed normally, or a modified LlmResponse to inject nudge.
        """
        # If tool was called, we're good - reset retry count and proceed
        if self._has_function_call(llm_response):
            session_id = self._get_session_id(callback_context)
            self._retry_counts[session_id] = 0
            return None

        # No tool call detected - check retry count
        session_id = self._get_session_id(callback_context)
        retry_count = self._retry_counts.get(session_id, 0)

        if retry_count >= self.max_retries:
            # Max retries exceeded - let it through (will fail gracefully)
            print(f"WARNING: LLM failed to use tool after {self.max_retries} attempts")
            return None

        # Increment retry count
        self._retry_counts[session_id] = retry_count + 1
        print(f"NUDGE ({retry_count + 1}/{self.max_retries}): LLM forgot to use tool, injecting reminder...")

        # =====================================================================
        # THE HACKY PART: Inject a nudge into the conversation
        # =====================================================================
        # Google ADK doesn't have a clean way to do this. We need to:
        # 1. Access the invocation context
        # 2. Add a message to the session/conversation
        # 3. Somehow trigger another model call
        #
        # The cleanest approach is to modify the response to include an error
        # that forces a retry, but this requires deep knowledge of ADK internals.
        #
        # For this example, we'll add the nudge to the session state and
        # handle retry at the application level (see extract_with_retry below).
        try:
            if hasattr(callback_context, "invocation_context"):
                ctx = callback_context.invocation_context
                # Store nudge in state for application-level retry handling
                if hasattr(ctx, "session") and ctx.session:
                    ctx.session.state["_needs_retry"] = True
                    ctx.session.state["_nudge_message"] = self.nudge_message
        except Exception as e:
            print(f"Could not set retry state: {e}")

        return None


# =============================================================================
# 3. CREATE AGENT WITH TOOL NUDGE
# =============================================================================


def create_extraction_agent() -> LlmAgent:
    """
    Create an agent configured for structured extraction with tool nudging.

    Note how much more setup this requires compared to Langroid:

    Langroid (3 lines in config):
        handle_llm_no_tool="You FORGOT to use the tool!"

    Google ADK (entire ToolNudgeCallback class + this function):
        after_model_callback=ToolNudgeCallback(...)
    """
    nudge_callback = ToolNudgeCallback(
        nudge_message="You MUST use the `extract_person_info` tool! Do not respond with plain text.",
        max_retries=3,
    )

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
        after_model_callback=nudge_callback,
    )

    return agent


# =============================================================================
# 4. RUN EXTRACTION WITH RETRY LOGIC
# =============================================================================
# Because Google ADK callbacks can't easily trigger retries, we need
# application-level retry logic. More boilerplate!


async def extract_with_retry(
    text: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Extract structured person information with retry on missing tool calls.

    This function handles the retry loop that Langroid does automatically
    with handle_llm_no_tool.

    Args:
        text: A passage containing information about a person.
        max_retries: Maximum number of retry attempts.

    Returns:
        A dictionary with extracted info, or None if extraction fails.
    """
    agent = create_extraction_agent()
    session_service = InMemorySessionService()

    for attempt in range(max_retries + 1):
        # Create fresh session for each attempt
        session = await session_service.create_session(
            app_name="extraction_demo",
            user_id="demo_user",
        )

        runner = Runner(
            agent=agent,
            app_name="extraction_demo",
            session_service=session_service,
        )

        # Build the message - include nudge on retries
        if attempt == 0:
            message_text = text
        else:
            # On retry, prepend the nudge to remind the LLM
            message_text = f"""
REMINDER: You MUST use the `extract_person_info` tool to extract information.
Do NOT respond with plain text. Use the tool now.

Here is the text to extract from:
{text}
"""

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
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "function_response") and part.function_response:
                        # Tool was called successfully
                        result = part.function_response.response
                        tool_was_used = True
                    elif hasattr(part, "function_call") and part.function_call:
                        # Tool is being called
                        tool_was_used = True

        if tool_was_used and result is not None:
            return result

        if attempt < max_retries:
            print(f"  Retry {attempt + 1}/{max_retries}: LLM didn't use tool, trying again...")

    print("  FAILED: LLM never used the tool after all retries")
    return None


def extract_person(text: str) -> dict | None:
    """Sync wrapper for the async extraction function."""
    return asyncio.run(extract_with_retry(text))


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

    print("=" * 70)
    print("GOOGLE ADK: Structured Information Extraction (with retry workaround)")
    print("=" * 70)
    print()
    print("This example shows the boilerplate needed to handle LLM forgetting tools:")
    print()
    print("  LANGROID (built-in, 1 line):")
    print('    handle_llm_no_tool = "You FORGOT to use the tool!"')
    print()
    print("  GOOGLE ADK (manual implementation):")
    print("    - ToolNudgeCallback class (~50 lines)")
    print("    - Application-level retry loop (~40 lines)")
    print("    - Session state management for retry tracking")
    print()

    for i, passage in enumerate(passages, 1):
        print(f"--- Passage {i} ---")
        print(passage.strip())
        print()

        result = extract_person(passage)

        if result:
            print(f"Extracted: {result}")
        else:
            print("Extraction failed!")
        print()


if __name__ == "__main__":
    main()
