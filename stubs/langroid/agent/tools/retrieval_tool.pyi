from langroid.agent.tool_message import ToolMessage as ToolMessage

class RetrievalTool(ToolMessage):
    request: str
    purpose: str
    query: str
    num_results: int
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
