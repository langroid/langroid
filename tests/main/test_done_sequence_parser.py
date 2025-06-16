"""Tests for done sequence DSL parser."""

import pytest

from langroid.agent.done_sequence_parser import (
    parse_done_sequence,
    parse_done_sequences,
)
from langroid.agent.task import AgentEvent, DoneSequence, EventType


def test_parse_simple_patterns():
    """Test parsing of simple single-letter patterns."""
    # Tool then Agent
    seq = parse_done_sequence("T, A")
    assert len(seq.events) == 2
    assert seq.events[0].event_type == EventType.TOOL
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE

    # LLM, Tool, Agent, LLM
    seq = parse_done_sequence("L, T, A, L")
    assert len(seq.events) == 4
    assert seq.events[0].event_type == EventType.LLM_RESPONSE
    assert seq.events[1].event_type == EventType.TOOL
    assert seq.events[2].event_type == EventType.AGENT_RESPONSE
    assert seq.events[3].event_type == EventType.LLM_RESPONSE

    # Without spaces
    seq = parse_done_sequence("T,A")
    assert len(seq.events) == 2
    assert seq.events[0].event_type == EventType.TOOL
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE


def test_parse_specific_tool():
    """Test parsing specific tool patterns."""
    seq = parse_done_sequence("T[calculator], A")
    assert len(seq.events) == 2
    assert seq.events[0].event_type == EventType.SPECIFIC_TOOL
    assert seq.events[0].tool_name == "calculator"
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE

    # Tool with hyphen
    seq = parse_done_sequence("T[my-tool], A")
    assert seq.events[0].tool_name == "my-tool"

    # Tool with dots
    seq = parse_done_sequence("T[com.example.tool], A")
    assert seq.events[0].tool_name == "com.example.tool"


def test_parse_content_match():
    """Test parsing content match patterns."""
    seq = parse_done_sequence("C[quit|exit]")
    assert len(seq.events) == 1
    assert seq.events[0].event_type == EventType.CONTENT_MATCH
    assert seq.events[0].content_pattern == "quit|exit"

    # Complex regex
    seq = parse_done_sequence("L, C[done.*complete]")
    assert len(seq.events) == 2
    assert seq.events[0].event_type == EventType.LLM_RESPONSE
    assert seq.events[1].event_type == EventType.CONTENT_MATCH
    assert seq.events[1].content_pattern == "done.*complete"


def test_parse_all_event_types():
    """Test all supported event types."""
    seq = parse_done_sequence("T, A, L, U, N")
    assert len(seq.events) == 5
    assert seq.events[0].event_type == EventType.TOOL
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE
    assert seq.events[2].event_type == EventType.LLM_RESPONSE
    assert seq.events[3].event_type == EventType.USER_RESPONSE
    assert seq.events[4].event_type == EventType.NO_RESPONSE


def test_parse_mixed_patterns():
    """Test complex mixed patterns."""
    seq = parse_done_sequence("T[search], A, T[calculator], A, C[complete]")
    assert len(seq.events) == 5
    assert seq.events[0].event_type == EventType.SPECIFIC_TOOL
    assert seq.events[0].tool_name == "search"
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE
    assert seq.events[2].event_type == EventType.SPECIFIC_TOOL
    assert seq.events[2].tool_name == "calculator"
    assert seq.events[3].event_type == EventType.AGENT_RESPONSE
    assert seq.events[4].event_type == EventType.CONTENT_MATCH
    assert seq.events[4].content_pattern == "complete"


def test_parse_existing_done_sequence():
    """Test that existing DoneSequence objects are returned unchanged."""
    original = DoneSequence(
        name="test",
        events=[
            AgentEvent(event_type=EventType.TOOL),
            AgentEvent(event_type=EventType.AGENT_RESPONSE),
        ],
    )

    result = parse_done_sequence(original)
    assert result is original  # Should be the same object


def test_parse_done_sequences_list():
    """Test parsing a mixed list of strings and DoneSequence objects."""
    sequences = parse_done_sequences(
        [
            "T, A",
            DoneSequence(
                name="existing", events=[AgentEvent(event_type=EventType.LLM_RESPONSE)]
            ),
            "T[calc], A",
            "C[quit]",
        ]
    )

    assert len(sequences) == 4
    assert all(isinstance(seq, DoneSequence) for seq in sequences)

    # Check first sequence (parsed from string)
    assert len(sequences[0].events) == 2
    assert sequences[0].events[0].event_type == EventType.TOOL

    # Check second sequence (existing object)
    assert sequences[1].name == "existing"
    assert len(sequences[1].events) == 1

    # Check third sequence (specific tool)
    assert sequences[2].events[0].event_type == EventType.SPECIFIC_TOOL
    assert sequences[2].events[0].tool_name == "calc"

    # Check fourth sequence (content match)
    assert sequences[3].events[0].event_type == EventType.CONTENT_MATCH
    assert sequences[3].events[0].content_pattern == "quit"


def test_parse_word_tokens():
    """Test parsing full word tokens like 'TOOL', 'AGENT'."""
    seq = parse_done_sequence("TOOL, AGENT")
    assert len(seq.events) == 2
    assert seq.events[0].event_type == EventType.TOOL
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE

    # Mixed case
    seq = parse_done_sequence("tool, Agent, LLM")
    assert len(seq.events) == 3
    assert seq.events[0].event_type == EventType.TOOL
    assert seq.events[1].event_type == EventType.AGENT_RESPONSE
    assert seq.events[2].event_type == EventType.LLM_RESPONSE


def test_parse_invalid_patterns():
    """Test error handling for invalid patterns."""
    with pytest.raises(ValueError, match="Invalid event token"):
        parse_done_sequence("T, X")  # X is not valid

    with pytest.raises(ValueError, match="Invalid event code with brackets"):
        parse_done_sequence("X[something]")  # X[] is not valid

    with pytest.raises(ValueError, match="No valid events found"):
        parse_done_sequence("")  # Empty pattern

    with pytest.raises(ValueError, match="No valid events found"):
        parse_done_sequence(",,,")  # Only commas

    with pytest.raises(ValueError, match="Expected string or DoneSequence"):
        parse_done_sequence(123)  # Wrong type


def test_parse_edge_cases():
    """Test edge cases in parsing."""
    # Extra spaces
    seq = parse_done_sequence("  T  ,  A  ")
    assert len(seq.events) == 2

    # Trailing comma
    seq = parse_done_sequence("T, A,")
    assert len(seq.events) == 2

    # Leading comma
    seq = parse_done_sequence(",T, A")
    assert len(seq.events) == 2
