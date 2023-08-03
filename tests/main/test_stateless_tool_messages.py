import itertools
from typing import Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global


class SquareTool(ToolMessage):
    request: str = "square"
    purpose: str = """
            To find the square of a <number> 
            """
    number: int

    def handle(self) -> str:
        """
        We are able to define the handler within the Tool class itself,
        rather than in the agent, since the tool does not require any
        member variables from the agent.
        We can think of these tools as "stateless" tools or Static tools,
        similar to static methods. Since the SquareTool is stateless,
        the corresponding agent method `square` can be automatically
        defined, using the body of the `handle` method.
        Thus there is no need to manually define a `square` method in the
        agent, as we normally would have to do for a (stateful) tool that has no
        `handle` method.
        See the `_get_tool_list` method in `agent/base.py` for how such
        tools are set up.
        """
        return str(self.number**2)


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
message_classes = [None, SquareTool]

# Get the cartesian product
cartesian_product = list(
    itertools.product(message_classes, use_vals, handle_vals, force_vals)
)

agent.enable_message(SquareTool)


@pytest.mark.parametrize(
    # cartesian product of all combinations of use, handle, force
    "msg_class, use, handle, force",
    cartesian_product,
)
def test_enable_message(
    msg_class: Optional[ToolMessage], use: bool, handle: bool, force: bool
):
    agent.enable_message(msg_class, use=use, handle=handle, force=force)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool in agent.llm_tools_map
        if msg_class is not None:
            assert agent.llm_tools_map[tool] == msg_class
            assert agent.llm_functions_map[tool] == msg_class.llm_function_schema()
        assert (tool in agent.llm_tools_handled) == handle
        assert (tool in agent.llm_tools_usable) == use
        assert (tool in agent.llm_functions_handled) == handle
        assert (tool in agent.llm_functions_usable) == use

    if msg_class is not None:
        assert (
            agent.llm_function_force is not None
            and agent.llm_function_force["name"] == tools[0]
        ) == force


@pytest.mark.parametrize("msg_class", [None, SquareTool])
def test_disable_message_handling(msg_class: Optional[ToolMessage]):
    agent.enable_message(SquareTool)
    agent.disable_message_handling(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool not in agent.llm_tools_handled
        assert tool not in agent.llm_functions_handled
        assert tool in agent.llm_tools_usable
        assert tool in agent.llm_functions_usable


@pytest.mark.parametrize("msg_class", [None, SquareTool])
def test_disable_message_use(msg_class: Optional[ToolMessage]):
    agent.enable_message(SquareTool)

    agent.disable_message_use(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool not in agent.llm_tools_usable
        assert tool not in agent.llm_functions_usable
        assert tool in agent.llm_tools_handled
        assert tool in agent.llm_functions_handled


NONE_MSG = "nothing to see here"

SQUARE_MSG = """
Ok, thank you.
{
"request": "square",
"number": 12
} 
Hope you can tell me!
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(SquareTool)
    assert agent.handle_message(NONE_MSG) is None
    assert agent.handle_message(SQUARE_MSG) == "144"

    agent.disable_message_handling(SquareTool)
    assert agent.handle_message(SQUARE_MSG) is None

    agent.disable_message_handling(SquareTool)
    assert agent.handle_message(SQUARE_MSG) is None

    agent.enable_message(SquareTool)
    assert agent.handle_message(SQUARE_MSG) == "144"


BAD_SQUARE_MSG = """
Ok, thank you.
{
"request": "square"
} 
Hope you can tell me!
"""


def test_handle_bad_tool_message():
    """
    Test that a correct tool name with bad/missing args is
            handled correctly, i.e. the agent returns a clear
            error message to the LLM so it can try to fix it.
    """
    agent.enable_message(SquareTool)
    assert agent.handle_message(NONE_MSG) is None
    result = agent.handle_message(BAD_SQUARE_MSG)
    assert all([x in result for x in ["square", "number", "required"]])


@pytest.mark.parametrize(
    "use_functions_api, message_class, prompt, result",
    [
        (
            False,
            SquareTool,
            "Start by asking me to square the number 9",
            "81",
        ),
        (
            True,
            SquareTool,
            "Start by asking me to square the number 9, using a tool/function-call",
            "81",
        ),
    ],
)
def test_llm_tool_message(
    test_settings: Settings,
    use_functions_api: bool,
    message_class: ToolMessage,
    prompt: str,
    result: str,
):
    """
    Test whether LLM is able to GENERATE message (tool) in required format, and the
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
    agent.enable_message(SquareTool)

    llm_msg = agent.llm_response_forget(prompt)
    tool_name = message_class.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], message_class)

    agent_result = agent.handle_message(llm_msg)
    assert result.lower() in agent_result.lower()
