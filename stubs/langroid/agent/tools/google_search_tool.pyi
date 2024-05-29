from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.parsing.web_search import google_search as google_search

class GoogleSearchTool(ToolMessage):
    request: str
    purpose: str
    query: str
    num_results: int
    def handle(self) -> str: ...
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
