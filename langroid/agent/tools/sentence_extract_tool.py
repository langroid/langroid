"""
A tool to extract sentence numbers from the last user message,
containing numbered sentences.

The idea is that when an LLM wants to (or is asked to) simply extract
portions of a message verbatim, it should use this tool/function to
SPECIFY what should be extracted, rather than actually extracting it.
The output will be in the form of a list of sentence numbers or ranges.
This will usually be much cheaper and faster than actually writing out the extracted
text. The handler of this tool/function will then extract the text and send it back.
"""

from typing import List

from langroid.agent.tool_message import ToolMessage


class SentenceExtractTool(ToolMessage):
    request: str = "extract_sentences"
    purpose: str = """
            To extract sentences from a body of text containing numbered 
            sentences, in the form of a <sentence_list> which is a list of sentence 
            numbers or ranges, like "10,12,14-17".
            """
    sentence_list: str

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [cls(sentence_list="1,3,5-7")]

    @classmethod
    def instructions(cls) -> str:
        return """
        You will use this tool/function to indicate certain sentences from 
        a body of text containing numbered sentences.
        """
