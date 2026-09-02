"""A tool that searches Google's organic results through SerpApi.

Set ``SERPAPI_API_KEY`` in the environment before using this tool.
"""

from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import serpapi_search


class SerpApiSearchTool(ToolMessage):
    request: str = "serpapi_search"
    purpose: str = """
            To search the web using SerpApi's Google engine and select up to
            <num_results> organic results returned for the given <query>. When
            using this tool, ONLY show the required JSON, DO NOT SAY ANYTHING
            ELSE. Wait for the results of the web search, and then use them
            to compose your response.
            """
    query: str
    num_results: int

    def handle(self) -> str:
        """Run the search and format its results for the agent."""
        search_results = serpapi_search(self.query, self.num_results)
        results_str = "\n\n".join(str(result) for result in search_results)
        return f"""
        BELOW ARE THE RESULTS FROM THE WEB SEARCH. USE THESE TO COMPOSE YOUR RESPONSE:
        {results_str}
        """

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(
                query="When was the Llama2 Large Language Model (LLM) released?",
                num_results=3,
            ),
        ]
