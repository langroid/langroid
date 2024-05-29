from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.parsing.web_search import duckduckgo_search as duckduckgo_search

class DuckduckgoSearchTool(ToolMessage):
    request: str
    purpose: str
    query: str
    num_results: int
    def handle(self) -> str: ...
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
