"""
A tool which returns a Search RAG response from the SciPhi API.
their titles, links, summaries. Since the tool is stateless (i.e. does not need
access to agent state), it can be enabled for any agent, without having to define a
special method inside the agent: `agent.enable_message(SciPhiSearchRAGTool)`

Example return output appears as follows below:

<-- Query -->
```
Find 3 results on the internet about the LK-99 superconducting material.
``

<-- Response (compressed for this example)-->
```
[ result1 ]

[ result2 ]

[ result3 ]

```

NOTE: Using this tool requires getting an API key from sciphi.ai.
Setup is as simple as shown below
# Get a free API key at https://www.sciphi.ai/account
# export SCIPHI_API_KEY=$MY_SCIPHI_API_KEY before running the agent
# OR add SCIPHI_API_KEY=$MY_SCIPHI_API_KEY to your .env file

This tool requires installing langroid with the `sciphi` extra, e.g.
`pip install langroid[sciphi]` or `poetry add langroid[sciphi]`
(it installs the `agent-search` package from pypi).

For more information, please refer to the official docs:
https://agent-search.readthedocs.io/en/latest/
"""

from typing import List

try:
    from agent_search import SciPhi
except ImportError:
    raise ImportError(
        "You are attempting to use the `agent-search` library;"
        "To use it, please install langroid with the `sciphi` extra, e.g. "
        "`pip install langroid[sciphi]` or `poetry add langroid[sciphi]` "
        "(it installs the `agent-search` package from pypi)."
    )

from langroid.agent.tool_message import ToolMessage


class SciPhiSearchRAGTool(ToolMessage):
    request: str = "web_search_rag"
    purpose: str = """
            To search the web with provider <search_provider> and 
            return a response summary with llm model <llm_model> the given <query>. 
            """
    query: str

    def handle(self) -> str:
        rag_response = SciPhi().get_search_rag_response(
            query=self.query, search_provider="bing", llm_model="SciPhi/Sensei-7B-V1"
        )
        result = rag_response["response"]
        result = (
            f"### RAG Response:\n{result}\n\n"
            + "### Related Queries:\n"
            + "\n".join(rag_response["related_queries"])
        )
        return result  # type: ignore

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                query="When was the Llama2 Large Language Model (LLM) released?",
            ),
        ]
