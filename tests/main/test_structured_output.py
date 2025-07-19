import copy
from typing import Any, Callable, List

import pytest
from pydantic import BaseModel, Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    ),
)
strict_cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
        supports_json_schema=True,
        supports_strict_tools=True,
        parallel_tool_calls=False,
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


@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_structured_output_list(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """
    Test whether LLM is able to GENERATE structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.config.use_tools_api = use_tools_api
    agent.enable_message(PresidentListTool)
    N = 3
    prompt = f"Show me examples of {N} Presidents of any set of countries you choose"
    llm_msg = agent.llm_response_forget(prompt)
    assert isinstance(agent.get_tool_messages(llm_msg)[0], PresidentListTool)
    agent_result = agent.agent_response(llm_msg)
    assert agent_result.content == str(N)


@pytest.mark.parametrize("use_functions_api", [False, True])
def test_llm_structured_output_nested(
    test_settings: Settings,
    use_functions_api: bool,
):
    """
    Test whether LLM is able to GENERATE nested structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(strict_cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.config.use_tools_api = True
    agent.enable_message(PresidentTool)
    country = "France"
    prompt = f"""
    Show me an example of a President of {country}.
    Make sure you use the `{PresidentTool.name()}` 
    correctly with ALL the required fields!
    """
    llm_msg = agent.llm_response_forget(prompt)
    assert isinstance(agent.get_tool_messages(llm_msg)[0], PresidentTool)
    assert country == agent.agent_response(llm_msg).content


@pytest.mark.parametrize("instructions", [False, True])
@pytest.mark.parametrize("use", [True, False])
@pytest.mark.parametrize("force_tools", [False, True])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_strict_json(
    instructions: bool,
    use: bool,
    force_tools: bool,
    use_tools_api: bool,
    use_functions_api: bool,
):
    """Tests structured output generation in strict JSON mode."""
    cfg = copy.deepcopy(strict_cfg)
    cfg.instructions_output_format = instructions
    cfg.use_output_format = use
    cfg.use_tools_on_output_format = force_tools
    cfg.use_tools = not use_functions_api
    cfg.use_functions_api = use_functions_api
    cfg.use_tools_api = use_tools_api
    agent = ChatAgent(cfg)

    def typed_llm_response(
        prompt: str,
        output_type: type,
    ) -> Any:
        response = agent[output_type].llm_response_forget(prompt)
        return agent.from_ChatDocument(response, output_type)

    def valid_typed_response(
        prompt: str,
        output_type: type,
        test: Callable[[Any], bool] = lambda _: True,
    ) -> bool:
        response = typed_llm_response(prompt, output_type)
        return isinstance(response, output_type) and test(response)

    president_prompt = "Show me an example of a President of France"
    presidents_prompt = "Show me an example of two Presidents"
    country_prompt = "Show me an example of a country"

    # The model always returns the correct type, even without instructions to do so
    assert valid_typed_response(president_prompt, President)
    assert valid_typed_response(president_prompt, PresidentTool)
    assert valid_typed_response(
        president_prompt,
        PresidentListTool,
        lambda output: len(output.my_presidents.presidents) == 1,
    )
    assert valid_typed_response(
        presidents_prompt,
        PresidentList,
        lambda output: len(output.presidents) == 2,
    )
    assert valid_typed_response(
        presidents_prompt,
        PresidentListTool,
        lambda output: len(output.my_presidents.presidents) == 2,
    )
    assert valid_typed_response(country_prompt, Country)

    # The model returns the correct type, even when the request is mismatched
    assert valid_typed_response(country_prompt, President)
    assert valid_typed_response(presidents_prompt, PresidentTool)
    assert valid_typed_response(country_prompt, PresidentList)
    assert valid_typed_response(president_prompt, Country)

    # Structured output handles simple Python types
    assert typed_llm_response("What is 2+2?", int) == 4
    assert typed_llm_response("Is 2+2 equal to 4?", bool)
    assert abs(typed_llm_response("What is the value of pi?", float) - 3.14) < 0.01
    assert valid_typed_response(president_prompt, str)


@pytest.mark.parametrize("instructions", [True, False])
@pytest.mark.parametrize("use", [True, False])
@pytest.mark.parametrize("force_tools", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.asyncio
async def test_llm_strict_json_async(
    instructions: bool,
    use: bool,
    force_tools: bool,
    use_tools_api: bool,
    use_functions_api: bool,
):
    """Tests asynchronous structured output generation in strict JSON mode."""
    cfg = copy.deepcopy(strict_cfg)
    cfg.instructions_output_format = instructions
    cfg.use_output_format = use
    cfg.use_tools_on_output_format = force_tools
    cfg.use_tools = not use_functions_api
    cfg.use_functions_api = use_functions_api
    cfg.use_tools_api = use_tools_api
    agent = ChatAgent(cfg)

    async def typed_llm_response(
        prompt: str,
        output_type: type,
    ) -> Any:
        response = await agent[output_type].llm_response_forget_async(prompt)
        return agent.from_ChatDocument(response, output_type)

    async def valid_typed_response(
        prompt: str,
        output_type: type,
        test: Callable[[Any], bool] = lambda _: True,
    ) -> bool:
        response = await typed_llm_response(prompt, output_type)
        return isinstance(response, output_type) and test(response)

    president_prompt = "Show me an example of a President of France"
    presidents_prompt = "Show me an example of two Presidents"
    country_prompt = "Show me an example of a country"

    # The model always returns the correct type, even without instructions to do so
    assert await valid_typed_response(president_prompt, President)
    assert await valid_typed_response(president_prompt, PresidentTool)
    assert await valid_typed_response(
        president_prompt,
        PresidentListTool,
        lambda output: len(output.my_presidents.presidents) == 1,
    )
    assert await valid_typed_response(
        presidents_prompt,
        PresidentList,
        lambda output: len(output.presidents) == 2,
    )
    assert await valid_typed_response(
        presidents_prompt,
        PresidentListTool,
        lambda output: len(output.my_presidents.presidents) == 2,
    )
    assert await valid_typed_response(country_prompt, Country)

    # The model returns the correct type, even when the request is mismatched
    assert await valid_typed_response(country_prompt, President)
    assert await valid_typed_response(presidents_prompt, PresidentTool)
    assert await valid_typed_response(country_prompt, PresidentList)
    assert await valid_typed_response(president_prompt, Country)

    # Structured output handles simple Python types
    assert await typed_llm_response("What is 2+2?", int) == 4
    assert await typed_llm_response("Is 2+2 equal to 4?", bool)
    assert (
        abs(await typed_llm_response("What is the value of pi?", float) - 3.14) < 0.01
    )
    assert await valid_typed_response(president_prompt, str)


@pytest.mark.parametrize("use", [True, False])
@pytest.mark.parametrize("handle", [True, False])
def test_output_format_tools(use: bool, handle: bool):
    cfg = copy.deepcopy(strict_cfg)
    cfg.handle_output_format = handle
    cfg.use_output_format = use
    agent = ChatAgent(cfg)

    agent_1 = agent[PresidentTool]
    agent_2 = agent[PresidentListTool]

    # agent[T] does not have T enabled for use or handling.
    for a in [agent, agent_1]:
        assert "president_list" not in a.llm_tools_usable
        assert "president_list" not in a.llm_tools_handled
    for a in [agent, agent_2]:
        assert "show_president" not in a.llm_tools_usable
        assert "show_president" not in a.llm_tools_handled

    agent.set_output_format(PresidentListTool)

    # setting the output format to T results in enabling use/handling of T
    # based on the cfg.use_output_format and cfg.handle_output_format
    assert ("president_list" in agent.llm_tools_handled) == handle
    assert ("president_list" in agent.llm_tools_usable) == use

    response = agent.llm_response_forget("Give me a list of presidents")
    # the response is handled only if cfg.handle_output_format is True
    assert (agent.handle_message(response) is not None) == handle

    agent.set_output_format(None)
    # We do not retain handling/use of
    # PresidentListTool as it was not explicitly enabled for handling/use
    # via `enable_message`.
    assert "president_list" not in agent.llm_tools_handled
    assert "president_list" not in agent.llm_tools_usable

    agent.set_output_format(PresidentTool, handle=True, use=True)
    assert "show_president" in agent.llm_tools_handled
    assert "show_president" in agent.llm_tools_usable

    response = agent.llm_response_forget("Give me a president")
    assert agent.handle_message(response) is not None

    # Explicitly enable PresidentTool
    agent.enable_message(PresidentTool)
    agent.set_output_format(PresidentListTool)

    # We DO retain the use/handling of PresidentTool
    # in the sets of enabled and handled tools
    # as it was explicitly enabled
    assert "show_president" in agent.llm_tools_handled
    assert "show_president" in agent.llm_tools_usable


@pytest.mark.parametrize("instructions", [True, False])
@pytest.mark.parametrize("use", [True, False])
def test_output_format_instructions(instructions: bool, use: bool):
    cfg = copy.deepcopy(strict_cfg)
    cfg.instructions_output_format = instructions
    cfg.use_output_format = use
    agent = ChatAgent(cfg)

    agent_1 = agent[PresidentTool]
    agent_2 = agent[PresidentListTool]
    # The strict-typed agent[T] will not have format instructions specifically for T
    for a in [agent, agent_1]:
        assert "president_list" not in a.output_format_instructions
    for a in [agent, agent_2]:
        assert "show_president" not in a.output_format_instructions

    agent.set_output_format(PresidentListTool)
    # We do add schema information to the instructions if the tool is enabled for use
    assert ("my_presidents" in agent.output_format_instructions) == (
        not use and instructions
    )
    # If we enable the tool for use, we only specify that the tool should be used
    assert ("`president_list`" in agent.output_format_instructions) == (
        use and instructions
    )
    # If the tool is enabled for use or instructions are generated, schema
    # information is added to the system message
    assert ("my_presidents" in agent._create_system_and_tools_message().content) == (
        use or instructions
    )

    agent.enable_message(PresidentTool)
    agent.set_output_format(PresidentTool)
    # The tool is already enabled and we do not add the schema to the
    # instructions
    assert ("`show_president`" in agent.output_format_instructions) == instructions
    assert "country" not in agent.output_format_instructions

    agent.set_output_format(Country)
    assert ("capital" in agent.output_format_instructions) == instructions

    agent.set_output_format(PresidentList, instructions=True)
    assert "presidents" in agent.output_format_instructions

    agent.set_output_format(None)
    assert agent.output_format_instructions == ""
