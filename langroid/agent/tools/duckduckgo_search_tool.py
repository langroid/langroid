"""
A tool to trigger a DuckDuckGo search for a given query, and return the top results with
their titles, links, summaries. Since the tool is stateless (i.e. does not need
access to agent state), it can be enabled for any agent, without having to define a
special method inside the agent: `agent.enable_message(DuckduckgoSearchTool)`
"""

from typing import List

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import duckduckgo_search


class DuckduckgoSearchTool(ToolMessage):
    request: str = "duckduckgo_search"
    purpose: str = """
            To search the web and return up to <num_results> 
            links relevant to the given <query>. When using this tool,
            ONLY show the required JSON, DO NOT SAY ANYTHING ELSE.
            Wait for the results of the web search, and then use them to
            compose your response.
            """
    query: str
    num_results: int

    def handle(self) -> str:
        """
        Conducts a search using DuckDuckGo based on the provided query
        and number of results by triggering a duckduckgo_search.

        Returns:
            str: A formatted string containing the titles, links, and
                summaries of each search result, separated by two newlines.
        """
        search_results = duckduckgo_search(self.query, self.num_results)
        # return Title, Link, Summary of each result, separated by two newlines
        results_str = "\n\n".join(str(result) for result in search_results)
        return f"""
        BELOW ARE THE RESULTS FROM THE WEB SEARCH. USE THESE TO COMPOSE YOUR RESPONSE:
        {results_str}
        """

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="When was the Llama2 Large Language Model (LLM) released?",
                num_results=3,
            ),
        ]
