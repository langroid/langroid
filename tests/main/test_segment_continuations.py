import pytest

from langroid.agent.special.relevance_extractor_agent import (
    RelevanceExtractorAgent,
    RelevanceExtractorAgentConfig,
)
from langroid.agent.tools.segment_extract_tool import SegmentExtractTool
from langroid.language_models.base import LLMMessage, Role
from langroid.parsing.utils import extract_numbered_segments, number_segments
from langroid.utils.constants import DONE


@pytest.mark.parametrize(
    "passage, spec, expected",
    [
        ("<#1#> First.\n\nContinuation.", "1", " First.\n\nContinuation."),
        ("<#1#> First.\n\nContinuation.", "2", ""),
        (
            "<#1#> First.\n\nContinuation.\n\nMore.",
            "1",
            " First.\n\nContinuation.\n\nMore.",
        ),
        (
            "<#1#> First.\n\nContinued. <#2#> Second.\n\nSecond continued.",
            "1",
            " First.\n\nContinued. ",
        ),
        (
            "<#1#> First.\n\nContinued. <#2#> Second.\n\nSecond continued.",
            "2",
            " Second.\n\nSecond continued.",
        ),
        (
            "Unnumbered intro.\n\n<#1#> First.\n \t\nContinued.\n\n<#2#> Second.",
            "1",
            " First.\n\nContinued.",
        ),
        (
            "<#1#> First.\n\n<#2#> Second.\n\nSecond continued.\n\n<#3#> Third.",
            "1,3",
            " First.\n\n Third.",
        ),
        ("No segment marker.\n\nMore text.", "1", ""),
        ("<#1#> First.\n\nMore text.", "", ""),
        ("<#1#> First. <#2#> Second.", "1,2", " First. ... Second."),
    ],
)
def test_extract_segment_paragraph_continuations(
    passage: str, spec: str, expected: str
) -> None:
    assert extract_numbered_segments(passage, spec) == expected


def test_whole_passage_segment_reaches_tool_result() -> None:
    passage = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    numbered = number_segments(passage, granularity=-1)
    assert extract_numbered_segments(numbered, "1").strip() == passage

    agent = RelevanceExtractorAgent(RelevanceExtractorAgentConfig(llm=None))
    agent.numbered_passage = numbered
    agent.message_history = [LLMMessage(role=Role.USER, content=numbered)]
    result = agent.extract_segments(SegmentExtractTool(segment_list="1"))
    assert result.removeprefix(DONE).strip() == passage
