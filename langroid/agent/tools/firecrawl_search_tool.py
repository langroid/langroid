"""
A tool to trigger a Firecrawl search for a given query and return the top results.
Since the tool is stateless (i.e. does not need
access to agent state), it can be enabled for any agent, without having to define a
special method inside the agent: `agent.enable_message(FirecrawlSearchTool)`

NOTE: To use this tool, you need to:

* set the FIRECRAWL_API_KEY environment variable in
your `.env` file, e.g. `FIRECRAWL_API_KEY=your_api_key_here`

* install langroid with the `firecrawl` extra, e.g.
`pip install langroid[firecrawl]` or `uv pip install langroid[firecrawl]`
or `poetry add langroid[firecrawl]` or `uv add langroid[firecrawl]`
(it installs the `firecrawl-py` package from pypi).

For more information, please refer to: https://www.firecrawl.dev/search?utm_source=langroid&utm_medium=integration
"""

from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import firecrawl_search


class FirecrawlSearchTool(ToolMessage):
    request: str = "firecrawl_search"
    purpose: str = """
            To search the web using Firecrawl and return up to <num_results>
            results relevant to the given <query>. When using this tool,
            ONLY show the required JSON, DO NOT SAY ANYTHING ELSE.
            Wait for the results of the web search, and then use them to
            compose your response.
            """
    query: str
    num_results: int

    def handle(self) -> str:
        """
        Conducts a search using the Firecrawl API based on the provided query
        and number of results by triggering a firecrawl_search.

        Returns:
            str: A formatted string containing the titles, links, and
                summaries of each search result, separated by two newlines.
        """

        search_results = firecrawl_search(self.query, self.num_results)
        # return Title, Link, Summary of each result, separated by two newlines
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
