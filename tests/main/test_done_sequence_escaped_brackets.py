from langroid.agent.done_sequence_parser import parse_done_sequence
from langroid.agent.task import EventType


def test_content_match_escaped_open_bracket_can_be_followed_by_event() -> None:
    sequence = parse_done_sequence(r"C[\[done,thanks], A")

    assert len(sequence.events) == 2
    assert sequence.events[0].event_type == EventType.CONTENT_MATCH
    assert sequence.events[0].content_pattern == r"\[done,thanks"
    assert sequence.events[1].event_type == EventType.AGENT_RESPONSE


def test_content_match_character_class_can_contain_commas() -> None:
    sequence = parse_done_sequence(r"C[[a,b]], A")

    assert len(sequence.events) == 2
    assert sequence.events[0].event_type == EventType.CONTENT_MATCH
    assert sequence.events[0].content_pattern == r"[a,b]"
    assert sequence.events[1].event_type == EventType.AGENT_RESPONSE


def test_content_match_escaped_closing_bracket_can_contain_commas() -> None:
    sequence = parse_done_sequence(r"C[done\],thanks], A")

    assert len(sequence.events) == 2
    assert sequence.events[0].event_type == EventType.CONTENT_MATCH
    assert sequence.events[0].content_pattern == r"done\],thanks"
    assert sequence.events[1].event_type == EventType.AGENT_RESPONSE
