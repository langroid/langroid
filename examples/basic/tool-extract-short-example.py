"""
Short example of using Langroid ToolMessage to extract structured info from a passage,
and perform computation on it.

Run like this:

python3 examples/basic/tool-extract-short-example.py

"""

import langroid as lr
from langroid.pydantic_v1 import BaseModel, Field
from langroid.agent.tools.orchestration import ResultTool


# desired output structure
class CompanyInfo(BaseModel):
    name: str = Field(..., description="name of company")
    shares: int = Field(..., description="shares outstanding of company")
    price: float = Field(..., description="price per share of company")


# tool definition based on this
class CompanyInfoTool(lr.agent.ToolMessage):
    request: str = "company_info_tool"  # agent method that handles this tool
    purpose: str = (
        "To extract <company_info> from a passage and compute market-capitalization."
    )
    company_info: CompanyInfo

    @classmethod
    def examples(cls):
        """Examples that will be compiled to few-shot examples for the LLM.
        Illustrating two types of examples below:
        - example instance
        - (thought, example) tuple
        """
        return [
            # Example 1: just the instance
            cls(company_info=CompanyInfo(name="IBM", shares=1.24e9, price=140.15)),
            # Example 2: (thought, instance) tuple
            (
                "I want to extract and present company info from the passage",
                cls(
                    company_info=CompanyInfo(name="Apple", shares=16.82e9, price=149.15)
                ),
            ),
        ]

    def handle(self) -> ResultTool:
        """Handle LLM's structured output if it matches CompanyInfo structure.
        This suffices for a "stateless" tool.
        If the tool handling requires agent state, then
        instead of this `handle` method, define a `company_info_tool`
        method in the agent.
        Since this method is returning a  ResultTool,
        the task of this agent will be terminated,
        with this tool T appearing in the result ChatDocument's `tool_messages` list.
        """
        mkt_cap = self.company_info.shares * self.company_info.price
        return ResultTool(
            market_cap=mkt_cap,
            info=self.company_info,
            comment="success",  # arbitrary undeclared fields allowed
        )


# define agent, attach the tool

agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        system_message="""
        Use the `company_info` tool to extract company information from a passage
        and compute market-capitalization.
        """,
    )
)

agent.enable_message(CompanyInfoTool)

# define and run task on a passage about some company

task = lr.Task(agent, interactive=False)
result = task.run(
    """
    Qualcomm has shares outstanding of 1.12 billion and a price per share of $217.09.
    """
)

# note the result.tool_messages will be a list containing
# an obj of type FinalResultTool, so we can extract fields from it.
company_result = result.tool_messages[0]
assert isinstance(company_result, ResultTool)
assert isinstance(company_result.info, CompanyInfo)

info = company_result.info
mktcap = company_result.market_cap
assert company_result.comment == "success"
print(
    f"""
    Found company info: {info} and market cap: {mktcap}
    """
)
