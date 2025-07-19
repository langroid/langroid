"""
Short example of using Langroid ToolMessage to extract structured info from a passage,
and perform computation on it.

Run like this (omit --model to default to GPT4o):

python3 examples/basic/tool-extract-short-example.py --model deepseek/deepseek-reasoner

or

uv run examples/basic/tool-extract-short-example.py --model deepseek/deepseek-reasoner

"""

from fire import Fire
from rich import print
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ResultTool
from pydantic import BaseModel, Field


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


def main(model: str = ""):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            system_message=f"""
            Use the TOOL `{CompanyInfoTool.name()}` 
            tool to extract company information from a passage
            and compute market-capitalization.
            """,
        )
    )

    agent.enable_message(CompanyInfoTool)

    # define and run task on a passage about some company

    task = lr.Task(agent, interactive=False)

    print(
        """
        [blue]Welcome to the company info extractor!
        Write a sentence containing company name, shares outstanding and share price,
        and the Agent will use a tool/function extract the info in structured form,
        and the tool-handler will compute the market-cap.[/blue]
        """
    )

    while True:
        statement = Prompt.ask(
            """
            Enter a sentence containing company name, 
            shares outstanding and share price, or 
            hit enter to use default sentence.
            """,
            default="""
            Qualcomm has shares outstanding of 1.12 billion and a 
            price per share of $217.09.
            """,
        )
        result = task.run(statement)
        if result is None:
            print("Tool-call failed, try again.")
            continue
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


if __name__ == "__main__":
    Fire(main)
