"""
Bare bones example of using tool/function-call

Run like this, optionally specifying an LLM:

python3 examples/basic/chat-tool-function.py

or

python3 examples/basic/chat-tool-function.py -m ollama/mistral:7b-instruct-v0.2-q8_0

"""


import langroid as lr
import langroid.language_models as lm
from pydantic import BaseModel, Field
import json
from fire import Fire

# define a nested structure for Company information


class CompanyFinancials(BaseModel):
    market_cap: float = Field(..., description="market capitalization of company")
    eps: float = Field(..., description="earnings per share of company")


class CompanyInfo(BaseModel):
    name: str = Field(..., description="name of company")
    industry: str = Field(..., description="industry of company")
    financials: CompanyFinancials = Field(..., description="financials of company")


# define a ToolMessage corresponding to the above structure


class CompanyInfoTool(lr.agent.ToolMessage):
    request: str = "company_info_tool"
    purpose: str = "To extract <company_info> from a given text passage"
    company_info: CompanyInfo

    def handle(self) -> str:
        """Handle LLM's structured output if it matches CompanyInfo structure.
        This suffices for a "stateless" tool.
        If the tool handling requires agent state, then
        instead of this `handle` method, define a `company_info_tool`
        method in the agent.
        """
        print(
            f"""
            DONE! Got Valid Company Info:
            {json.dumps(self.company_info.dict(), indent=4)}
            """
        )


def run(model: str = ""):  # or, e.g., "ollama/mistral:7b-instruct-v0.2-q8_0"
    lm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,  # or
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
        It has a market capitalization of 2.5 trillion dollars and an earnings 
        per share of 5.68.
        """

    # see that the LLM extracts the company information and presents it using the tool
    response = agent.llm_response(paragraph)

    print(response.content)

    # wrap the agent in a Task, so that the ToolMessage is handled

    task = lr.Task(agent, interactive=False)
    task.run(paragraph, turns=2)


if __name__ == "__main__":
    Fire(run)
