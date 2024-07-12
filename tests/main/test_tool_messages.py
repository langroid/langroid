import itertools
import json
import random
from typing import List, Literal, Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.parse_json import extract_top_level_json
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.pydantic_v1 import BaseModel, Field
from langroid.utils.configuration import Settings, set_global


class CountryCapitalMessage(ToolMessage):
    request: str = "country_capital"
    purpose: str = "To check whether <city> is the capital of <country>."
    country: str = "France"
    city: str = "Paris"

    @classmethod
    def examples(cls) -> List["CountryCapitalMessage"]:
        # illustrating two types of examples
        return [
            (
                "Need to check if Paris is the capital of France",
                cls(country="France", city="Paris"),
            ),
            cls(country="France", city="Marseille"),
        ]


class FileExistsMessage(ToolMessage):
    request: str = "file_exists"
    purpose: str = "To check whether a certain <filename> is in the repo."
    filename: str = Field(..., description="File name to check existence of")

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md"),
            cls(filename="Dockerfile"),
        ]


class PythonVersionMessage(ToolMessage):
    request: str = "python_version"
    purpose: str = "To check which version of Python is needed."

    @classmethod
    def examples(cls) -> List["PythonVersionMessage"]:
        return [
            cls(),
        ]


DEFAULT_PY_VERSION = "3.9"


class MessageHandlingAgent(ChatAgent):
    def file_exists(self, message: FileExistsMessage) -> str:
        return "yes" if message.filename == "requirements.txt" else "no"

    def python_version(self, PythonVersionMessage) -> str:
        return DEFAULT_PY_VERSION

    def country_capital(self, message: CountryCapitalMessage) -> str:
        return (
            "yes" if (message.city == "Paris" and message.country == "France") else "no"
        )


cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
    system_message="""
    VERY IMPORTANT: IF you see a possibility of using a tool/function,
    you MUST use it, and MUST NOT ASK IN NATURAL LANGUAGE.
    """,
)
agent = MessageHandlingAgent(cfg)

# Define the range of values each variable can have
use_vals = [True, False]
handle_vals = [True, False]
force_vals = [True, False]
message_classes = [None, FileExistsMessage, PythonVersionMessage]

# Get the cartesian product
cartesian_product = list(
    itertools.product(message_classes, use_vals, handle_vals, force_vals)
)

agent.enable_message(FileExistsMessage)
agent.enable_message(PythonVersionMessage)


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


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
def test_disable_message_handling(msg_class: Optional[ToolMessage]):
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

    agent.disable_message_handling(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool not in agent.llm_tools_handled
        assert tool not in agent.llm_functions_handled
        assert tool in agent.llm_tools_usable
        assert tool in agent.llm_functions_usable


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
def test_disable_message_use(msg_class: Optional[ToolMessage]):
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

    agent.disable_message_use(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool not in agent.llm_tools_usable
        assert tool not in agent.llm_functions_usable
        assert tool in agent.llm_tools_handled
        assert tool in agent.llm_functions_handled


@pytest.mark.parametrize("msg_cls", [PythonVersionMessage, FileExistsMessage])
def test_usage_instruction(msg_cls: ToolMessage):
    usage = msg_cls.usage_examples()
    jsons = extract_top_level_json(usage)
    assert all(
        json.loads(j)["request"] == msg_cls.default_value("request") for j in jsons
    )


NONE_MSG = "nothing to see here"

FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists",
"filename": "test.txt"
} 
Hope you can tell me!
"""

PYTHON_VERSION_MSG = """
great, please tell me this --
{
"request": "python_version"
}/if you know it
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(NONE_MSG) is None
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"

    agent.disable_message_handling(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"

    agent.disable_message_handling(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"


BAD_FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists"
} 
Hope you can tell me!
"""


def test_handle_bad_tool_message():
    """
    Test that a correct tool name with bad/missing args is
            handled correctly, i.e. the agent returns a clear
            error message to the LLM so it can try to fix it.
    """
    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(NONE_MSG) is None
    result = agent.handle_message(BAD_FILE_EXISTS_MSG)
    assert "file_exists" in result and "filename" in result and "required" in result


@pytest.mark.parametrize(
    "use_functions_api",
    [True, False],
)
@pytest.mark.parametrize(
    "message_class, prompt, result",
    [
        (
            FileExistsMessage,
            "You have to find out whether the file 'requirements.txt' exists",
            "yes",
        ),
        (
            PythonVersionMessage,
            "Find out about the python version",
            "3.9",
        ),
        (
            CountryCapitalMessage,
            "You have to check whether Paris is the capital of France",
            "yes",
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
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    if not agent.llm.is_openai_chat_model() and use_functions_api:
        pytest.skip(
            f"""
            Function Calling not available for {agent.config.llm.chat_model}: skipping
            """
        )

    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(CountryCapitalMessage)

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


def test_llm_non_tool(test_settings: Settings):
    agent = MessageHandlingAgent(cfg)
    llm_msg = agent.llm_response_forget(
        "Ask me to check what is the population of France."
    ).content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result is None


# Test that malformed tool messages results in proper err msg
class NumPair(BaseModel):
    xval: int
    yval: int


class NabroskiTool(ToolMessage):
    request: str = "nabroski"
    purpose: str = "to request computing the Nabroski transform of <num_pair>"
    num_pair: NumPair

    def handle(self) -> str:
        return str(3 * self.num_pair.xval + self.num_pair.yval)


wrong_nabroski_tool = """
{
"request": "nabroski",
"num_pair": {
    "xval": 1
    }
}
"""


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_agent_malformed_tool(
    test_settings: Settings,
    use_functions_api: bool,
):
    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    response = agent.agent_response(wrong_nabroski_tool)
    assert "num_pair" in response.content and "yval" in response.content


class EulerTool(ToolMessage):
    request: str = "euler"
    purpose: str = "to request computing the Euler transform of <num_pair>"
    num_pair: NumPair

    def handle(self) -> str:
        return str(2 * self.num_pair.xval - self.num_pair.yval)


class GaussTool(ToolMessage):
    request: str = "gauss"
    purpose: str = "to request computing the Gauss transform of (<x>, <y>)"
    xval: int
    yval: int

    def handle(self) -> str:
        return str((self.xval + self.yval) * self.yval)


class CoinFlipTool(ToolMessage):
    request: str = "coin_flip"
    purpose: str = "to request a random coin flip"

    def handle(self) -> Literal["Heads", "Tails"]:
        heads = random.random() > 0.5
        return "Heads" if heads else "Tails"


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_agent_infer_tool(
    test_settings: Settings,
    use_functions_api: bool,
):
    set_global(test_settings)
    gauss_request = """{"xval": 1, "yval": 3}"""
    nabrowski_or_euler_request = """{"num_pair": {"xval": 1, "yval": 3}}"""
    euler_request = """{"request": "euler", "num_pair": {"xval": 1, "yval": 3}}"""
    additional_args_request = """{"xval": 1, "yval": 3, "zval": 4}"""
    additional_args_request_specified = """
    {"request": "gauss", "xval": 1, "yval": 3, "zval": 4}
    """
    no_args_request = """{}"""
    no_args_request_specified = """{"request": "coin_flip"}"""

    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    agent.enable_message(GaussTool)
    agent.enable_message(CoinFlipTool)
    agent.enable_message(EulerTool, handle=False)

    # Nabrowski is the only option prior to enabling EulerTool handling
    assert agent.agent_response(nabrowski_or_euler_request).content == "6"

    # Enable handling EulerTool, this makes nabrowski_or_euler_request ambiguous
    agent.enable_message(EulerTool)

    # Gauss is the only option
    assert agent.agent_response(gauss_request).content == "12"

    # Explicit requests are forwarded to the correct handler
    assert agent.agent_response(euler_request).content == "-1"

    # We cannot infer the correct tool if there exist multiple matches
    assert agent.agent_response(nabrowski_or_euler_request) is None

    # We do not infer tools where the request has additional arguments
    assert agent.agent_response(additional_args_request) is None
    # But additional args are acceptable when the tool is specified
    assert agent.agent_response(additional_args_request_specified).content == "12"

    # We do not infer tools with no args
    assert agent.agent_response(no_args_request) is None
    # Request must be specified
    assert agent.agent_response(no_args_request_specified).content in ["Heads", "Tails"]


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_no_llm_response(
    test_settings: Settings,
    use_functions_api: bool,
):
    """Test that agent.llm_response does not respond to tool messages."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    nabroski_tool = NabroskiTool(num_pair=NumPair(xval=1, yval=2)).to_json()
    response = agent.llm_response(nabroski_tool)
    assert response is None
