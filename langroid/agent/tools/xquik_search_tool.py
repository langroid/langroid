"""
A tool to search public X posts with Xquik and return matching posts.
Since the tool is stateless, it can be enabled for any agent without
defining a handler method: `agent.enable_message(XquikSearchTool)`.

NOTE: To use this tool, set the XQUIK_API_KEY environment variable in
your `.env` file, e.g. `XQUIK_API_KEY=your_api_key_here`.
"""

from typing import List, Tuple

from langroid.agent.tool_message import ToolMessage
from langroid.parsing.web_search import xquik_search


class XquikSearchTool(ToolMessage):
    request: str = "xquik_search"
    purpose: str = """
            To search public X posts using Xquik and return up to <num_results>
            posts relevant to the given <query>. The query can include X search
            operators such as from:user, #hashtags, exact phrases, since:YYYY-MM-DD,
            until:YYYY-MM-DD, and min_faves:N. When using this tool, ONLY show
            the required JSON, DO NOT SAY ANYTHING ELSE. Wait for the X post
            results, and then use them to compose your response.
            """
    query: str
    num_results: int

    def handle(self) -> str:
        """
        Conducts an Xquik search based on the query and result count.

        Returns:
            str: A formatted string containing each post's author or id,
                URL, and text, separated by two newlines.
        """
        search_results = xquik_search(self.query, self.num_results)
        results_str = "\n\n".join(str(result) for result in search_results)
        return f"""
        BELOW ARE THE RESULTS FROM THE X POST SEARCH.
        USE THESE TO COMPOSE YOUR RESPONSE:
        {results_str}
        """

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(
                query='from:xquikcom "API"',
                num_results=3,
            ),
        ]
