from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage


class RetrievalTool(ToolMessage):
    """
    Retrieval tool, only to be used by a DocChatAgent.
    The handler method is defined in DocChatAgent.retrieval_tool
    """

    request: str = "retrieval_tool"
    purpose: str = """
            To retrieve up to <num_results> passages from a document-set, that are 
            relevant to a <query>, which could be a question or simply a topic or 
            search phrase. 
            """
    query: str
    num_results: int

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(
                query="What are the eligibility criteria for the scholarship?",
                num_results=3,
            ),
            cls(
                query="Self-Attention mechanism in RNNs",
                num_results=5,
            ),
        ]
