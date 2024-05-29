from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.parsing.web_search import metaphor_search as metaphor_search

class MetaphorSearchTool(ToolMessage):
    request: str
    purpose: str
    query: str
    num_results: int
    def handle(self) -> str: ...
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
