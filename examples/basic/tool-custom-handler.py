"""
Short example of using `_handler` attribute in ToolMessage to define
custom name for `Agent` tool handler.

Run like this:

python3 examples/basic/tool-custom-handler.py

"""

import requests

import langroid as lr
from langroid.pydantic_v1 import Field


class CountryLanguageTool(lr.agent.ToolMessage):
    request: str = "country_language_tool"
    purpose: str = "To determine <language> spoken in specific country."
    country_name: str = Field(..., description="country name")
    _handler: str = "country_tools_handler"


class CountryPopulationTool(lr.agent.ToolMessage):
    request: str = "country_population_tool"
    purpose: str = "To determine <population> of specific country."
    country_name: str = Field(..., description="country name")
    _handler: str = "country_tools_handler"


class CountryAreaTool(lr.agent.ToolMessage):
    request: str = "country_area_tool"
    purpose: str = "To determine <area> of specific country."
    country_name: str = Field(..., description="country name")
    _handler: str = "country_tools_handler"


class AssistantAgent(lr.ChatAgent):
    def country_tools_handler(self, tool: lr.agent.ToolMessage):
        response = requests.get(
            f"https://restcountries.com/v3.1/name/{tool.country_name}", timeout=5
        )
        if not response.ok:
            return "invalid country name"

        try:
            data = response.model_dump_json()[0]
        except (ValueError, IndexError):
            return "invalid response"

        match tool.request:
            case "country_language_tool":
                language = ", ".join(data["languages"].values())
                return language
            case "country_population_tool":
                population_millions = data["population"] / 1e6
                return f"{population_millions:.1f} million people"
            case "country_area_tool":
                area_sq_km = data["area"] / 1e6
                return f"{area_sq_km:.1f} million sq. km"

        return "invalid tool name"


def make_assistant_task() -> lr.Task:
    llm_config = lr.language_models.OpenAIGPTConfig(
        temperature=0.2, max_output_tokens=250
    )

    assistant_config = lr.ChatAgentConfig(
        system_message="""
        You are a helpful assistant helping users with country-related questions.

        You know answers to the following questions:
          - what is the <language> spoken in specific country?
          - what is <population> of specific country?
          - what is <areay> of specific country?

        Ask user for the country name and information that he is interested in.
        Then use the appropriate tool to find the answer.
        """,
        llm=llm_config,
    )

    assistant_agent = AssistantAgent(assistant_config)
    assistant_agent.enable_message(CountryLanguageTool)
    assistant_agent.enable_message(CountryPopulationTool)
    assistant_agent.enable_message(CountryAreaTool)

    assistant_task = lr.Task(agent=assistant_agent, interactive=True)
    return assistant_task


if __name__ == "__main__":
    task = make_assistant_task()
    task.run()
