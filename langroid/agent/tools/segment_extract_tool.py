"""
A tool to extract segment numbers from the last user message,
containing numbered segments.

The idea is that when an LLM wants to (or is asked to) simply extract
portions of a message verbatim, it should use this tool/function to
SPECIFY what should be extracted, rather than actually extracting it.
The output will be in the form of a list of segment numbers or ranges.
This will usually be much cheaper and faster than actually writing out the extracted
text. The handler of this tool/function will then extract the text and send it back.
"""

from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage


class SegmentExtractTool(ToolMessage):
    request: str = "extract_segments"
    purpose: str = """
            To extract segments from a body of text containing numbered 
            segments, in the form of a <segment_list> which is a list of segment 
            numbers or ranges, like "10,12,14-17".
            """
    segment_list: str

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            (
                "I want to extract segments 1, 3, and 5 thru 7",
                cls(segment_list="1,3,5-7"),
            )
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        Use this tool/function to indicate certain segments from 
        a body of text containing numbered segments.
        """
