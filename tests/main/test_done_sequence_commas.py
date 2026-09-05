from langroid.agent.done_sequence_parser import parse_done_sequence
from langroid.agent.task import EventType


def test_content_match_pattern_can_contain_commas() -> None:
    sequence = parse_done_sequence(r"L, C[done,\s*thanks]")

    assert len(sequence.events) == 2
    assert sequence.events[0].event_type == EventType.LLM_RESPONSE
    assert sequence.events[1].event_type == EventType.CONTENT_MATCH
    assert sequence.events[1].content_pattern == r"done,\s*thanks"
