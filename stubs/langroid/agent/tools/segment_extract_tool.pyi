from langroid.agent.tool_message import ToolMessage as ToolMessage

class SegmentExtractTool(ToolMessage):
    request: str
    purpose: str
    segment_list: str
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
    @classmethod
    def instructions(cls) -> str: ...
