"""
Bare bones example of using tool/function-call

Run like this, optionally specifying an LLM:

python3 examples/basic/chat-tool-function.py

or

python3 examples/basic/chat-tool-function.py -m ollama/mistral:7b-instruct-v0.2-q8_0

or 

uv run examples/basic/chat-tool-function.py -m deepseek/deepseek-reasoner

"""

from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import FinalResultTool
from langroid.pydantic_v1 import BaseModel, Field

# define a nested structure for Company information


class CompanyFinancials(BaseModel):
    shares: int = Field(..., description="shares outstanding of company")
    price: float = Field(..., description="price per share of company")
    eps: float = Field(..., description="earnings per share of company")


class CompanyInfo(BaseModel):
    name: str = Field(..., description="name of company")
    industry: str = Field(..., description="industry of company")
    financials: CompanyFinancials = Field(..., description="financials of company")


# define a ToolMessage corresponding to the above structure


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
        - (description, example) tuple
        """
        return [
            cls(
                company_info=CompanyInfo(
                    name="IBM",
                    industry="Technology",
                    financials=CompanyFinancials(shares=1.24e9, price=140.15, eps=4.68),
                )
            ),
            (
                "I want to extract and present company info from the passage",
                cls(
                    company_info=CompanyInfo(
                        name="Apple",
                        industry="Technology",
                        financials=CompanyFinancials(
                            shares=16.82e9, price=149.15, eps=5.68
                        ),
                    )
                ),
            ),
        ]

    def handle(self) -> FinalResultTool:
        """Handle LLM's structured output if it matches CompanyInfo structure.
        This suffices for a "stateless" tool.
        If the tool handling requires agent state, then
        instead of this `handle` method, define a `company_info_tool`
        method in the agent.
        """
        mkt_cap = (
            self.company_info.financials.shares * self.company_info.financials.price
        )
        print(
            f"""
            Got Valid Company Info.
            The market cap of {self.company_info.name} is ${mkt_cap/1e9}B.
            """
        )
        return FinalResultTool(
            market_cap=mkt_cap,
            info=self.company_info,
        )


def run(model: str = ""):  # or, e.g., "ollama/mistral:7b-instruct-v0.2-q8_0"
    lm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,  # or
    )
    tool_name = CompanyInfoTool.default_value("request")
    agent_config = lr.ChatAgentConfig(
        llm=lm_config,
        system_message=f"""
        You are a company-info extraction expert. When user gives you a TEXT PASSAGE,
        simply extract the company information and 
        present it using the `{tool_name}` tool/function-call.
        """,
    )
    agent = lr.ChatAgent(agent_config)
    agent.enable_message(CompanyInfoTool)

    # text to present to the LLM
    paragraph = """
        Apple Inc. is an American multinational technology company that specializes in 
        consumer electronics, computer software, and online services.
        It has shares outstanding of 16.82 billion, and a price per share of $149.15.
        The earnings per share is $5.68.
        """

    # test 1:
    # see that the LLM extracts the company information and presents it using the tool
    response = agent.llm_response(paragraph)

    print(response.content)

    # test 2:
    # wrap the agent in a Task, so that the ToolMessage is handled by the handle method
    task = lr.Task(agent, interactive=False)
    result = task[FinalResultTool].run(paragraph)
    assert result.market_cap > 0
    assert "Apple" in result.info.name


if __name__ == "__main__":
    Fire(run)
