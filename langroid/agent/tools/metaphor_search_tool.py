"""
A tool to trigger a Metaphor search for a given query,
(https://docs.exa.ai/reference/getting-started)
and return the top results with their titles, links, summaries.
Since the tool is stateless (i.e. does not need
access to agent state), it can be enabled for any agent, without having to define a
special method inside the agent: `agent.enable_message(MetaphorSearchTool)`

NOTE: To use this tool, you need to:

* set the METAPHOR_API_KEY environment variables in
your `.env` file, e.g. `METAPHOR_API_KEY=your_api_key_here`
(Note as of 28 Jan 2023, Metaphor renamed to Exa, so you can also use
`EXA_API_KEY=your_api_key_here`)

* install langroid with the `metaphor` extra, e.g.
`pip install langroid[metaphor]` or `poetry add langroid[metaphor]`
(it installs the `metaphor-python` package from pypi).

For more information, please refer to the official docs:
https://metaphor.systems/
"""

from typing import List

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import metaphor_search


class MetaphorSearchTool(ToolMessage):
    request: str = "metaphor_search"
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
        Conducts a search using the metaphor API based on the provided query
        and number of results by triggering a metaphor_search.

        Returns:
            str: A formatted string containing the titles, links, and
                summaries of each search result, separated by two newlines.
        """

        search_results = metaphor_search(self.query, self.num_results)
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
