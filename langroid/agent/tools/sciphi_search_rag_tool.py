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

<-- Response -->
```
The LK-99 superconducting material has captured the internet's attention due to its purported ability to conduct electricity without resistance at temperatures as high as room temperature [11]. This claim, if true, would represent a groundbreaking advance in materials science, potentially revolutionizing fields such as quantum computing and magnetic levitation [10]. However, the validity of LK-99 as a superconductor has been called into question by several studies. Researchers at the National Physics Laboratory in India found that replicates of LK-99 did not exhibit superconductivity but only diamagnetism, suggesting that the original claims may have been misinterpreted [4]. Similarly, a study by scientists in China and Japan, who were among the first to report superconductivity in LK-99, has been met with skepticism, with some experts arguing that the observed effects could be explained by other physical phenomena [15].  

The debate intensified as new data emerged, with some researchers reporting that the material does not show superconductivity at room temperature but might do so under different conditions [13]. This has led to a flurry of activity in the scientific community, with 16 teams racing to validate the superconducting properties of LK-99, albeit with mixed results [15]. The original team behind the discovery of LK-99 has maintained its position, promising to provide more evidence in a forthcoming publication [20]. In summary, while the initial claims about LK-99 have generated significant excitement, the scientific consensus remains divided, and further research is necessary to determine the true nature of this material.
```

NOTE: Using this tool requires getting an API key from sciphi.ai.
Setup is as simple as shown below
# Get a free API key at https://www.sciphi.ai/account
# export SCIPHI_API_KEY=$MY_SCIPHI_API_KEY

For more information, please refer to the official docs:
https://agent-search.readthedocs.io/en/latest/
"""

from agent_search import SciPhi

from langroid.agent.tool_message import ToolMessage


class SciPhiSearchRAGTool(ToolMessage):
    request: str = "web_search_rag"
    purpose: str = """
            To search the web with provider <search_provider> and return a response summary
            with llm model <llm_model> the given <query>. 
            """
    query: str
    search_provider: str = "bing"  # bing or agent-search
    include_related_queries: bool = True
    llm_model: str = "SciPhi/Sensei-7B-V1"
    recursive_mode: bool = True

    def handle(self) -> str:
        rag_response = SciPhi().get_search_rag_response(
            query=self.query,
            search_provider=self.search_provider,
            llm_model=self.llm_model,
        )
        result = rag_response["response"]
        if self.include_related_queries:
            result = (
                f"### RAG Response:\n{result}\n\n"
                + "### Related Queries:\n"
                + "\n".join(rag_response["related_queries"])
            )
        return result
