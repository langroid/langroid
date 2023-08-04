import itertools

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.stateless_tools.google_search_tool import GoogleSearchTool
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
)
agent = ChatAgent(cfg)

# Define the range of values each variable can have
use_vals = [True, False]
handle_vals = [True, False]
force_vals = [True, False]
message_classes = [None, GoogleSearchTool]

# Get the cartesian product
cartesian_product = list(
    itertools.product(message_classes, use_vals, handle_vals, force_vals)
)

agent.enable_message(GoogleSearchTool)


NONE_MSG = "nothing to see here"

SEARCH_MSG = """
Ok, thank you.
{
"request": "web_search",
"query": "wikipedia american independence",
"num_results": 3
} 
Hope you can tell me!
"""


def test_agent_handle_message():
    """
    Test whether the agent handles tool messages correctly,
    when these are manually generated.
    """
    agent.enable_message(GoogleSearchTool)
    assert agent.handle_message(NONE_MSG) is None
    assert len(agent.handle_message(SEARCH_MSG).split("\n\n")) == 3


BAD_SEARCH_MSG = """
Ok, thank you.
{
"request": "web_search"
} 
Hope you can tell me!
"""


def test_handle_bad_tool_message():
    """
    Test that a correct tool name with bad/missing args is
            handled correctly, i.e. the agent returns a clear
            error message to the LLM so it can try to fix it.
    """
    agent.enable_message(GoogleSearchTool)
    assert agent.handle_message(NONE_MSG) is None
    result = agent.handle_message(BAD_SEARCH_MSG)
    assert all(
        [x in result for x in ["web_search", "query", "num_results", "required"]]
    )


@pytest.mark.parametrize(
    "use_functions_api, message_class, prompt",
    [
        (
            False,
            GoogleSearchTool,
            "Find 3 results on the internet about the LK-99 superconductor",
        ),
        (
            True,
            GoogleSearchTool,
            "Find 3 results on the internet about the LK-99 superconductor",
        ),
    ],
)
def test_llm_tool_message(
    test_settings: Settings,
    use_functions_api: bool,
    message_class: ToolMessage,
    prompt: str,
):
    """
    Test whether LLM is able to GENERATE message (tool) in required format, AND the
    agent handles the message correctly.
    Args:
        test_settings: test settings from conftest.py
        use_functions_api: whether to use LLM's functions api or not
            (i.e. use the langroid ToolMessage tools instead).
        message_class: the message class (i.e. tool/function) to test
        prompt: the prompt to use to induce the LLM to use the tool
        result: the expected result from agent handling the tool-message
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(GoogleSearchTool)

    llm_msg = agent.llm_response_forget(prompt)
    tool_name = message_class.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], message_class)

    agent_result = agent.handle_message(llm_msg)
    assert len(agent_result.split("\n\n")) == 3
    assert all("LK-99" in x for x in agent_result.split("\n\n"))
