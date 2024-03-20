"""
A tool to trigger a Google search for a given query, and return the top results with
their titles, links, summaries. Since the tool is stateless (i.e. does not need
access to agent state), it can be enabled for any agent, without having to define a
special method inside the agent: `agent.enable_message(GoogleSearchTool)`

NOTE: Using this tool requires setting the GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).
"""

from typing import List

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import google_search


class GoogleSearchTool(ToolMessage):
    request: str = "web_search"
    purpose: str = """
            To search the web and return up to <num_results> links relevant to 
            the given <query>. 
            """
    query: str
    num_results: int

    def handle(self) -> str:
        search_results = google_search(self.query, self.num_results)
        # return Title, Link, Summary of each result, separated by two newlines
        return "\n\n".join(str(result) for result in search_results)

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="When was the Llama2 Large Language Model (LLM) released?",
                num_results=3,
            ),
        ]
