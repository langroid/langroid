"""Parser for done sequence DSL (Domain Specific Language).

Converts string patterns into DoneSequence objects for convenient task completion
configuration.

Examples:
    "T, A" -> Tool followed by Agent response
    "T[calculator], A" -> Specific tool 'calculator' followed by Agent response
    "L, T, A, L" -> LLM, Tool, Agent, LLM sequence
    "C[quit|exit]" -> Content matching regex pattern
"""

import re
from typing import List, Union

from .task import AgentEvent, DoneSequence, EventType


def parse_done_sequence(sequence: Union[str, DoneSequence]) -> DoneSequence:
    """Parse a string pattern or return existing DoneSequence unchanged.

    Args:
        sequence: Either a DoneSequence object or a string pattern to parse

    Returns:
        DoneSequence object

    Raises:
        ValueError: If the string pattern is invalid
    """
    if isinstance(sequence, DoneSequence):
        return sequence

    if not isinstance(sequence, str):
        raise ValueError(f"Expected string or DoneSequence, got {type(sequence)}")

    events = _parse_string_pattern(sequence)
    return DoneSequence(events=events)


def _parse_string_pattern(pattern: str) -> List[AgentEvent]:
    """Parse a string pattern into a list of AgentEvent objects.

    Pattern format:
        - Single letter codes: T, A, L, U, N, C
        - Specific tools: T[tool_name]
        - Content match: C[regex_pattern]
        - Separated by commas, spaces allowed

    Args:
        pattern: String pattern to parse

    Returns:
        List of AgentEvent objects

    Raises:
        ValueError: If pattern is invalid
    """
    events = []

    # Split by comma and strip whitespace
    parts = [p.strip() for p in pattern.split(",")]

    for part in parts:
        if not part:
            continue

        event = _parse_event_token(part)
        events.append(event)

    if not events:
        raise ValueError(f"No valid events found in pattern: {pattern}")

    return events


def _parse_event_token(token: str) -> AgentEvent:
    """Parse a single event token into an AgentEvent.

    Args:
        token: Single event token (e.g., "T", "T[calc]", "C[quit|exit]")

    Returns:
        AgentEvent object

    Raises:
        ValueError: If token is invalid
    """
    # Check for bracket notation
    bracket_match = re.match(r"^([A-Z])\[([^\]]+)\]$", token)

    if bracket_match:
        event_code = bracket_match.group(1)
        param = bracket_match.group(2)

        if event_code == "T":
            # Specific tool: T[tool_name]
            return AgentEvent(event_type=EventType.SPECIFIC_TOOL, tool_name=param)
        elif event_code == "C":
            # Content match: C[regex_pattern]
            return AgentEvent(event_type=EventType.CONTENT_MATCH, content_pattern=param)
        else:
            raise ValueError(
                f"Invalid event code with brackets: {event_code}. "
                "Only T[tool] and C[pattern] are supported."
            )

    # Simple single-letter codes
    event_map = {
        "T": EventType.TOOL,
        "A": EventType.AGENT_RESPONSE,
        "L": EventType.LLM_RESPONSE,
        "U": EventType.USER_RESPONSE,
        "N": EventType.NO_RESPONSE,
        "C": EventType.CONTENT_MATCH,  # C without brackets matches any content
    }

    if token in event_map:
        return AgentEvent(event_type=event_map[token])

    # If not a single letter, could be a full event type name
    token_upper = token.upper()
    if token_upper == "TOOL":
        return AgentEvent(event_type=EventType.TOOL)
    elif token_upper == "AGENT":
        return AgentEvent(event_type=EventType.AGENT_RESPONSE)
    elif token_upper == "LLM":
        return AgentEvent(event_type=EventType.LLM_RESPONSE)
    elif token_upper == "USER":
        return AgentEvent(event_type=EventType.USER_RESPONSE)
    else:
        raise ValueError(
            f"Invalid event token: '{token}'. "
            "Valid tokens are: T, A, L, U, N, C, or T[tool_name], C[pattern]"
        )


def parse_done_sequences(
    sequences: List[Union[str, DoneSequence]]
) -> List[DoneSequence]:
    """Parse a list of mixed string patterns and DoneSequence objects.

    Args:
        sequences: List containing strings and/or DoneSequence objects

    Returns:
        List of DoneSequence objects
    """
    return [parse_done_sequence(seq) for seq in sequences]
