from typing import List

import pytest
from pydantic import BaseModel, Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
)


class Country(BaseModel):
    """Info about a country"""

    name: str = Field(..., description="Name of the country")
    capital: str = Field(..., description="Capital of the country")


class President(BaseModel):
    """Info about a president of a country"""

    country: Country = Field(..., description="Country of the president")
    name: str = Field(..., description="Name of the president")
    election_year: int = Field(..., description="Year of election of the president")


class PresidentList(BaseModel):
    """List of presidents of various countries"""

    presidents: List[President] = Field(..., description="List of presidents")


class PresidentListTool(ToolMessage):
    """Tool/Function-call to present a list of presidents"""

    request: str = "president_list"
    purpose: str = """To show a list of presidents"""
    my_presidents: PresidentList = Field(..., description="List of presidents")

    def handle(self) -> str:
        return str(len(self.my_presidents.presidents))

    @classmethod
    def examples(cls) -> List["PresidentListTool"]:
        """Examples to use in prompt; Not essential, but increases chance of LLM
        generating in the expected format"""
        return [
            cls(
                my_presidents=PresidentList(
                    presidents=[
                        President(
                            country=Country(name="USA", capital="Washington DC"),
                            name="Joe Biden",
                            election_year=2020,
                        ),
                        President(
                            country=Country(name="France", capital="Paris"),
                            name="Emmanuel Macron",
                            election_year=2017,
                        ),
                    ]
                )
            ),
        ]


class PresidentTool(ToolMessage):
    """Tool/function to generate a president example"""

    request: str = "show_president"
    purpose: str = """To generate an example of a president"""
    president: President = Field(..., description="An example of a president")

    def handle(self) -> str:
        return self.president.country.name

    @classmethod
    def examples(cls) -> List["PresidentTool"]:
        """Examples to use in prompt; Not essential, but increases chance of LLM
        generating in the expected format"""
        return [
            cls(
                president=President(
                    name="Joe Biden",
                    country=Country(name="USA", capital="Washington DC"),
                    election_year=2020,
                )
            )
        ]


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_structured_output_list(
    test_settings: Settings,
    use_functions_api: bool,
):
    """
    Test whether LLM is able to GENERATE structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(PresidentListTool)
    N = 3
    prompt = f"Show me examples of {N} Presidents of any set of countries you choose"
    llm_msg = agent.llm_response_forget(prompt)
    tool_name = PresidentListTool.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call is not None
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], PresidentListTool)

    agent_result = agent.agent_response(llm_msg)
    assert agent_result.content == str(N)


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_structured_output_nested(
    test_settings: Settings,
    use_functions_api: bool,
):
    """
    Test whether LLM is able to GENERATE structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(PresidentTool)
    country = "France"
    prompt = f"Show me an example of a President of {country}"
    llm_msg = agent.llm_response_forget(prompt)
    tool_name = PresidentTool.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call is not None
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], PresidentTool)

    assert country == agent.agent_response(llm_msg).content
