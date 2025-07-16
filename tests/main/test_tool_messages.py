import itertools
import json
import random
from typing import Any, List, Literal, Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools import DonePassTool, DoneTool
from langroid.agent.tools.orchestration import (
    AgentDoneTool,
    FinalResultTool,
    ResultTool,
)
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMFunctionSpec,
    LLMMessage,
    OpenAIJsonSchemaSpec,
    OpenAIToolCall,
    OpenAIToolSpec,
    Role,
)
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parse_json import extract_top_level_json
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.pydantic_v1 import BaseModel, Field
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.utils.types import is_callable


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
    purpose: str = """
    To check whether a certain <filename> is in the repo,
    recursively if needed, as specified by <recurse>.
    """
    filename: str = Field(..., description="File name to check existence of")
    recurse: bool = Field(..., description="Whether to recurse into subdirectories")

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md", recurse=True),
            cls(filename="Dockerfile", recurse=False),
        ]


class PythonVersionMessage(ToolMessage):
    request: str = "python_version"
    _handler: str = "tool_handler"
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

    def tool_handler(self, message: ToolMessage) -> str:
        if message.request == "python_version":
            return DEFAULT_PY_VERSION
        else:
            return "invalid tool name"

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


def test_tool_message_name():
    assert FileExistsMessage.default_value("request") == FileExistsMessage.name()


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
@pytest.mark.parametrize("use", [True, False])
@pytest.mark.parametrize("handle", [True, False])
@pytest.mark.parametrize("force", [True, False])
def test_enable_message(
    msg_class: Optional[ToolMessage], use: bool, handle: bool, force: bool
):
    agent.enable_message(msg_class, use=use, handle=handle, force=force)
    usable_tools = agent.llm_tools_usable
    tools = agent._get_tool_list(msg_class)
    for tool in set(tools).intersection(usable_tools):
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
    agent.enable_message([FileExistsMessage, PythonVersionMessage])
    usable_tools = agent.llm_tools_usable.copy()

    agent.disable_message_handling(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in set(tools).intersection(usable_tools):
        assert tool not in agent.llm_tools_handled
        assert tool not in agent.llm_functions_handled
        assert tool in agent.llm_tools_usable
        assert tool in agent.llm_functions_usable


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
def test_disable_message_use(msg_class: Optional[ToolMessage]):
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    usable_tools = agent.llm_tools_usable.copy()

    agent.disable_message_use(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in set(tools).intersection(usable_tools):
        assert tool not in agent.llm_tools_usable
        assert tool not in agent.llm_functions_usable
        assert tool in agent.llm_tools_handled
        assert tool in agent.llm_functions_handled

    # check that disabling tool-use works as expected:
    # Tools part of sys msg should be updated, and
    # LLM should not be able to use this tool
    agent.disable_message_use(FileExistsMessage)
    agent.disable_message_use(PythonVersionMessage)
    response = agent.llm_response_forget("Is there a README.md file?")
    assert agent.get_tool_messages(response) == []


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
"filename": "test.txt",
"recurse": true
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
    assert agent.handle_message(FILE_EXISTS_MSG).content == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG).content == "3.9"

    agent.disable_message_handling(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG).content == "3.9"

    agent.disable_message_handling(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG).content == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG).content == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG).content == "3.9"


BAD_FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists"
}
Hope you can tell me!
"""


@pytest.mark.parametrize("as_string", [False, True])
def test_handle_bad_tool_message(as_string: bool):
    """
    Test that a correct tool name with bad/missing args is
            handled correctly, i.e. the agent returns a clear
            error message to the LLM so it can try to fix it.

    as_string: whether to pass the bad tool message as a string or as an LLM msg
    """
    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(NONE_MSG) is None
    if as_string:
        # set up a prior LLM-originated msg, to mock a scenario
        # where the last msg was from LLM, prior to calling
        # handle_message with the bad tool message -- we are trying to
        # test that the error is raised correctly in this case
        agent.llm_response("3+4=")
        result = agent.handle_message(BAD_FILE_EXISTS_MSG)
    else:
        bad_tool_from_llm = agent.create_llm_response(BAD_FILE_EXISTS_MSG)
        result = agent.handle_message(bad_tool_from_llm)
    assert "file_exists" in result and "filename" in result and "required" in result


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "use_functions_api",
    [True, False],
)
@pytest.mark.parametrize(
    "use_tools_api",
    [True],  # ONLY test tools-api since OpenAI has deprecated functions-api
)
@pytest.mark.parametrize(
    "message_class, prompt, result",
    [
        (
            FileExistsMessage,
            """
            You have to find out whether the file 'requirements.txt' exists in the repo,
            recursively exploring subdirectories if needed.
            """,
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
    use_tools_api: bool,
    message_class: ToolMessage,
    prompt: str,
    result: str,
    stream: bool,
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
    cfg.llm.stream = stream
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.config.use_tools_api = use_tools_api

    agent.enable_message(
        [
            FileExistsMessage,
            PythonVersionMessage,
            CountryCapitalMessage,
        ]
    )

    llm_msg = agent.llm_response_forget(prompt)
    tool_name = message_class.name()
    if use_functions_api:
        if use_tools_api:
            assert llm_msg.oai_tool_calls[0].function.name == tool_name
        else:
            assert llm_msg.function_call.name == tool_name

    tools = agent.get_tool_messages(llm_msg)
    assert len(tools) == 1
    assert isinstance(tools[0], message_class)

    agent_result = agent.handle_message(llm_msg).content

    assert result.lower() in agent_result.lower()


def test_llm_non_tool(test_settings: Settings):
    """Having no tools enabled should result in a None handle_message result"""
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


class CoriolisTool(ToolMessage):
    """Tool for testing handling of optional arguments, with default values."""

    request: str = "coriolis"
    purpose: str = "to request computing the Coriolis transform of <cats> and <cows>"
    cats: int
    cows: int = 5

    def handle(self) -> str:
        # same as NabroskiTool result
        return str(3 * self.cats + self.cows)


wrong_nabroski_tool = """
{
"request": "nabroski",
"num_pair": {
    "xval": 1
    }
}
"""


@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("strict_recovery", [True, False])
@pytest.mark.parametrize("as_string", [True, False])
def test_agent_malformed_tool(
    test_settings: Settings,
    use_tools_api: bool,
    use_functions_api: bool,
    stream: bool,
    strict_recovery: bool,
    as_string: bool,
):
    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
        strict_recovery=strict_recovery,
    )
    cfg.llm.stream = stream
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    if as_string:
        # set up a prior LLM-originated msg, to mock a scenario
        # where the last msg was from LLM, prior to calling
        # handle_message with the bad tool message -- we are trying to
        # test that the error is raised correctly in this case
        agent.llm_response("3+4=")
        response = agent.agent_response(wrong_nabroski_tool)
    else:
        bad_tool_from_llm = agent.create_llm_response(wrong_nabroski_tool)
        response = agent.agent_response(bad_tool_from_llm)
    # We expect an error msg containing certain specific field names
    assert "num_pair" in response.content and "yval" in response.content


class FruitPair(BaseModel):
    pears: int
    apples: int


class EulerTool(ToolMessage):
    request: str = "euler"
    purpose: str = "to request computing the Euler transform of <fruit_pair>"
    fruit_pair: FruitPair

    def handle(self) -> str:
        return str(2 * self.fruit_pair.pears - self.fruit_pair.apples)


class BoilerTool(ToolMessage):
    request: str = "boiler"
    purpose: str = "to request computing the Boiler transform of <fruit_pair>"
    fruit_pair: FruitPair

    def handle(self) -> str:
        return str(3 * self.fruit_pair.pears - 5 * self.fruit_pair.apples)


class SumTool(ToolMessage):
    request: str = "sum"
    purpose: str = "to request computing the sum of <x> and <y>"
    x: int
    y: int

    def handle(self) -> str:
        return str(self.x + self.y)


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


@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_agent_infer_tool(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    set_global(test_settings)
    gauss_request = """{"xval": 1, "yval": 3}"""
    boiler_or_euler_request = """{"fruit_pair": {"pears": 1, "apples": 3}}"""
    euler_request = """{"request": "euler", "fruit_pair": {"pears": 1, "apples": 3}}"""
    additional_args_request = """{"xval": 1, "yval": 3, "zval": 4}"""
    additional_args_request_specified = """
    {"request": "gauss", "xval": 1, "yval": 3, "zval": 4}
    """
    no_args_request = """{}"""
    no_args_request_specified = """{"request": "coin_flip"}"""

    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(
        [
            NabroskiTool,
            GaussTool,
            CoinFlipTool,
            BoilerTool,
        ]
    )
    agent.enable_message(EulerTool, handle=False)

    # Boiler is the only option prior to enabling EulerTool handling
    assert agent.agent_response(boiler_or_euler_request).content == "-12"

    # Enable handling EulerTool, this makes nabrowski_or_euler_request ambiguous
    agent.enable_message(EulerTool)
    agent.enable_message(BoilerTool)

    # Gauss is the only option
    assert agent.agent_response(gauss_request).content == "12"

    # Explicit requests are forwarded to the correct handler
    assert agent.agent_response(euler_request).content == "-1"

    # We cannot infer the correct tool if there exist multiple matches
    assert agent.agent_response(boiler_or_euler_request) is None

    # We do not infer tools where the request has additional arguments
    assert agent.agent_response(additional_args_request) is None
    # But additional args are acceptable when the tool is specified
    assert agent.agent_response(additional_args_request_specified).content == "12"

    # We do not infer tools with no args
    assert agent.agent_response(no_args_request) is None
    # Request must be specified
    assert agent.agent_response(no_args_request_specified).content in ["Heads", "Tails"]


@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_no_llm_response(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """Test that agent.llm_response does not respond to tool messages."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    nabroski_tool = NabroskiTool(num_pair=NumPair(xval=1, yval=2)).to_json()
    response = agent.llm_response(nabroski_tool)
    assert response is None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_no_task(
    test_settings: Settings,
    use_functions_api: bool,
    stream: bool,
):
    """Test tool handling without running task, i.e. directly using
    agent.llm_response and agent.agent_response methods."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
    )
    cfg.llm.stream = stream
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool, use=True, handle=True)

    response = agent.llm_response("What is Nabroski of 1 and 2?")
    assert isinstance(agent.get_tool_messages(response)[0], NabroskiTool)
    result = agent.agent_response(response)
    assert result.content == "5"


@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_optional_args(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """Test that ToolMessage where some args are optional (i.e. have default values)
    works well, i.e. LLM is able to generate all args if needed, including optionals."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)

    agent.enable_message(CoriolisTool, use=True, handle=True)
    response = agent.llm_response("What is the Coriolis transform of 1, 2?")
    assert isinstance(agent.get_tool_messages(response)[0], CoriolisTool)
    tool = agent.get_tool_messages(response)[0]
    assert tool.cats == 1 and tool.cows == 2


@pytest.mark.parametrize("tool", [NabroskiTool, CoriolisTool])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_tool_task(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
    stream: bool,
    tool: ToolMessage,
):
    """
    Test "full life cycle" of tool, when using Task.run().

    1. invoke LLM api with tool-spec
    2. LLM generates tool
    3. ChatAgent.agent_response handles tool, result added to ChatAgent msg history
    5. invoke LLM api with tool result
    """

    set_global(test_settings)
    llm_config = OpenAIGPTConfig(max_output_tokens=3_000, timeout=120)
    cfg = ChatAgentConfig(
        llm=llm_config,
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
        system_message=f"""
        You will be asked to compute a certain transform of two numbers,
        using a tool/function-call that you have access to.
        When you receive the answer from the tool, say {DONE} and show the answer.
        DO NOT SAY {DONE} until you receive a specific result from the tool.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(tool)
    task = Task(agent, interactive=False)

    request = tool.default_value("request")
    result = task.run(f"What is the {request} transform of 3 and 5?")
    assert "14" in result.content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_multi_tool(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
    stream: bool,
):
    """
    Test "full life cycle" of tool, when using Task.run().

    1. invoke LLM api with tool-spec
    2. LLM generates tool
    3. ChatAgent.agent_response handles tool, result added to ChatAgent msg history
    5. invoke LLM api with tool result
    """

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
        system_message=f"""
        You will be asked to compute transforms of two numbers,
        using tools/function-calls that you have access to.
        When you are asked for MULTIPLE transforms, you MUST
        use MULTIPLE tools/functions.
        When you receive the answers from the tools, say {DONE} and show the answers.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    agent.enable_message(GaussTool)
    task = Task(agent, interactive=False)

    # First test without task; using individual methods
    # ---

    result = task.run(
        """
        Compute these:
        (A) Nabroski transform of 3 and 5
        (B) Gauss transform of 1 and 2
        """
    )
    # Nabroski: 3*3 + 5 = 14
    # Gauss: (1+2)*2 = 6
    assert "14" in result.content and "6" in result.content


@pytest.mark.parametrize("stream", [False, True])
def test_oai_tool_choice(
    test_settings: Settings,
    stream: bool,
):
    """
    Test tool_choice for OpenAI-like LLM APIs.
    """

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=False,  # langroid tools
        use_functions_api=True,  # openai tools/fns
        use_tools_api=True,  # openai tools/fns
        system_message=f"""
        You will be asked to compute an operation or transform of two numbers,
        either using your own knowledge, or
        using a tool/function-call that you have access to.
        When you have an answer, say {DONE} and show the answer.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(SumTool)

    chat_doc = agent.create_user_response("What is the sum of 3 and 5?")
    chat_doc.oai_tool_choice = "auto"
    response = agent.llm_response_forget(chat_doc)

    # expect either SumTool or direct result without tool
    assert "8" in response.content or isinstance(
        agent.get_tool_messages(response)[0], SumTool
    )

    chat_doc = agent.create_user_response("What is the double of 5?")
    chat_doc.oai_tool_choice = "none"
    response = agent.llm_response_forget(chat_doc)
    assert "10" in response.content

    chat_doc = agent.create_user_response("What is the sum of 3 and 5?")
    chat_doc.oai_tool_choice = "required"
    response = agent.llm_response_forget(chat_doc)
    assert isinstance(agent.get_tool_messages(response)[0], SumTool)

    agent.enable_message(NabroskiTool, force=True)
    response = agent.llm_response("What is the nabroski of 3 and 5?")
    assert "nabroski" in response.content.lower() or isinstance(
        agent.get_tool_messages(response)[0], NabroskiTool
    )


@pytest.mark.parametrize(
    "result_type",
    [
        "final_tool",
        "result_tool",
        "agent_done",
        "tool",
        "int",
        "list",
        "dict",
        "ChatDocument",
        "pydantic",
    ],
)
@pytest.mark.parametrize(
    "tool_handler", ["notool", "handle", "response", "response_with_doc"]
)
def test_tool_handlers_and_results(result_type: str, tool_handler: str):
    """Test various types of ToolMessage handlers, and check that they can
    return arbitrary result types"""

    class SpecialResult(BaseModel):
        """To illustrating returning an arbitrary Pydantic object as a result"""

        answer: int
        details: str = "nothing"

    def result_fn(x: int) -> Any:
        match result_type:
            case "int":
                return x + 5
            case "dict":
                return {"answer": x + 5, "details": "something"}
            case "list":
                return [x + 5, x * 2]
            case "ChatDocument":
                return ChatDocument(
                    content=str(x + 5),
                    metadata=ChatDocMetaData(sender="Agent"),
                )
            case "pydantic":
                return SpecialResult(answer=x + 5)
            case "tool":
                # return tool, to be handled by sub-task
                return UberTool(x=x)
            case "result_tool":
                return ResultTool(answer=x + 5)
            case "final_tool":
                return FinalResultTool(
                    special=SpecialResult(answer=x + 5),  # explicitly declared
                    # arbitrary new fields that were not declared in the class...
                    extra_special=SpecialResult(answer=x + 10),
                    # ... does not need to be a Pydantic object
                    arbitrary_obj=dict(answer=x + 15),
                )
            case "agent_done":
                # pass on to parent, to handle with UberTool,
                # which is NOT enabled for this agent
                return AgentDoneTool(tools=[UberTool(x=x)])

    class UberTool(ToolMessage):
        request: str = "uber_tool"
        purpose: str = "to request the 'uber' transform of a  number <x>"
        x: int

        def handle(self) -> Any:
            return FinalResultTool(answer=self.x + 5)

    class CoolToolWithHandle(ToolMessage):
        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def handle(self) -> Any:
            return result_fn(self.x)

    class MyAgent(ChatAgent):
        def init_state(self) -> None:
            super().init_state()
            self.state: int = 100
            self.sender: str = ""
            self.llm_sent: bool = False

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            self.llm_sent = True
            return super().llm_response(message)

        def handle_message_fallback(
            self, msg: str | ChatDocument
        ) -> str | ChatDocument | None:
            """Handle non-tool LLM response"""
            if self.llm_sent:
                x = int(msg.content)
                return result_fn(x)

    class CoolToolWithResponse(ToolMessage):
        """To test that `response` handler works as expected,
        and is able to read and modify agent state.
        """

        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def response(self, agent: MyAgent) -> Any:
            agent.state += 1
            return result_fn(self.x)

    class CoolToolWithResponseDoc(ToolMessage):
        """
        To test that `response` handler works as expected,
        is able to read and modify agent state, and
        when using a `chat_doc` argument, and is able to read values from it.
        """

        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def response(self, agent: MyAgent, chat_doc: ChatDocument) -> Any:
            agent.state += 1
            agent.sender = chat_doc.metadata.sender
            return result_fn(self.x)

    match tool_handler:
        case "handle":
            tool_class = CoolToolWithHandle
        case "response":
            tool_class = CoolToolWithResponse
        case "response_with_doc":
            tool_class = CoolToolWithResponseDoc
        case "notool":
            tool_class = None

    agent = MyAgent(
        ChatAgentConfig(
            name="Test",
            # no need for a real LLM, use a mock
            llm=MockLMConfig(
                # mock LLM generating a CoolTool variant
                response_fn=lambda x: (
                    tool_class(x=int(x)).model_dump_json()
                    if tool_class is not None
                    else x
                ),
            ),
        )
    )
    if tool_class is not None:
        agent.enable_message(tool_class)

    tool_result = result_type in ["final_tool", "agent_done", "tool", "result_tool"]
    task = Task(
        agent,
        interactive=False,
        # need to specify task done when result is not FinalResultTool
        done_if_response=[] if tool_result else [Entity.AGENT],
    )
    result = task.run("3")
    if tool_handler == "response":
        assert agent.state == 101
    if tool_handler == "response_with_doc":
        assert agent.state == 101
        assert agent.sender == "LLM"

    if not tool_result:
        # CoolTool handler returns a non-tool result containing 8, and
        # we terminate task on agent_response, via done_if_response,
        # so the result.content == 8
        assert "8" in result.content
    elif result_type == "result_tool":
        # CoolTool handler/response returns a ResultTool containing answer == 8
        tool = result.tool_messages[0]
        assert isinstance(tool, ResultTool)
        assert tool.answer == 8
    else:
        # When CoolTool handler returns a ToolMessage,
        # test that it is handled correctly by sub-task or a parent.

        another_agent = ChatAgent(
            ChatAgentConfig(
                name="Another",
                llm=MockLMConfig(response_fn=lambda x: x),  # pass thru
            )
        )
        another_agent.enable_message(UberTool)
        another_task = Task(another_agent, interactive=False)
        another_task.add_sub_task(task)
        result = another_task.run("3")

        if result_type == "final_tool":
            # task's CoolTool handler returns FinalResultTool
            # which short-circuits parent task and returns as a tool
            # in tool_messages list of the final result
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert isinstance(tool.special, SpecialResult)
            assert tool.special.answer == 8
            assert tool.extra_special.answer == 13
            assert tool.arbitrary_obj["answer"] == 18
        elif result_type == "agent_done":
            # inner task's CoolTool handler returns a DoneTool containing
            # UberTool, which is handled by the parent "another_agent"
            # which returns a FinalResultTool containing answer == 8
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert tool.answer == 8

            # Now disable parent agent's handling of UberTool
            another_agent.disable_message_handling(UberTool)
            # another_task = Task(another_agent, interactive=False)
            # another_task.add_sub_task(task)
            result = another_task.run("3")
            # parent task is unable to handle UberTool, so will stall and return None
            assert result is None
            another_agent.enable_message(UberTool)

        elif result_type == "tool":
            # inner Task CoolTool handler returns UberTool (with NO done signal),
            # which it is unable to handle, so stalls and returns None,
            # and so does parent another_task
            assert result is None

            # Now reverse it: make another_task a sub-task of task, and
            # test handling UberTool returned by task handler, by sub-task another_task
            another_task = Task(another_agent, interactive=False)
            # task = Task(agent, interactive=False)
            task.add_sub_task(another_task)
            result = task.run("3")
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert tool.answer == 8

            another_agent.disable_message_handling(UberTool)
            result = task.run("3")
            # subtask stalls, parent stalls, returns None
            assert result is None


@pytest.mark.parametrize("llm_tool", ["pair", "final_tool"])
@pytest.mark.parametrize("handler_result_type", ["agent_done", "final_tool"])
@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
def test_llm_end_with_tool(
    handler_result_type: str,
    llm_tool: str,
    use_fn_api: bool,
    use_tools_api: bool,
):
    """
    Test that an LLM can directly or indirectly trigger task-end, and return a Tool as
    result. There are 3 ways:
    - case llm_tool == "final_tool":
        LLM returns a Tool (llm_tool == "final_tool") derived from FinalResultTool,
        with field(s) containing a structured Pydantic object -- in this case the task
        ends immediately without any agent response handling the tool
    - case llm_tool == "pair":
        LLM returns a PairTool, which is handled by the agent, which returns either
        - AgentDoneTool, with `tools` field set to [self], or
        - FinalResultTool, with `result` field set to the PairTool
    """

    class Pair(BaseModel):
        a: int
        b: int

    class PairTool(ToolMessage):
        """Handle the LLM-generated tool, signal done or final-result and
        return it as the result."""

        request: str = "pair_tool"
        purpose: str = "to return a <pair> of numbers"
        pair: Pair

        def handle(self) -> Any:
            if handler_result_type == "final_tool":
                # field name can be anything; `result` is just an example.
                return FinalResultTool(result=self)
            else:
                return AgentDoneTool(tools=[self])

    class FinalResultPairTool(FinalResultTool):
        request: str = "final_result_pair_tool"
        purpose: str = "Present final result <pair>"
        pair: Pair
        _allow_llm_use: bool = True

    final_result_pair_tool_name = FinalResultPairTool.default_value("request")

    class MyAgent(ChatAgent):
        def init_state(self) -> None:
            super().init_state()
            self.numbers: List[int] = []

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            """Mock human user input: they start with 0, then increment by 1"""
            last_num = self.numbers[-1] if self.numbers else 0
            new_num = last_num + 1
            self.numbers.append(new_num)
            return self.create_user_response(content=str(new_num))

    pair_tool_name = PairTool.default_value("request")

    if llm_tool == "pair":
        # LLM generates just PairTool , to be handled by its tool handler
        system_message = f"""
            Ask the user for their next number.
            Once you have collected 2 distinct numbers, present these as a pair
            using the TOOL: `{pair_tool_name}`.
            """
    else:
        system_message = f"""
            Ask the user for their next number.
            Once you have collected 2 distinct numbers, present these as the
            final result using the TOOL: `{final_result_pair_tool_name}`.
        """

    agent = MyAgent(
        ChatAgentConfig(
            name="MyAgent",
            system_message=system_message,
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
        )
    )
    if llm_tool == "pair":
        agent.enable_message(PairTool)
    else:
        agent.enable_message(FinalResultPairTool)

    # we are mocking user response, so need to set only_user_quits_root=False
    # so that the done signal (AgentDoneTool or FinalResultTool) actually end the task.
    task = Task(agent, interactive=True, only_user_quits_root=False)
    result = task.run()
    tool = result.tool_messages[0]
    if llm_tool == "pair":
        if handler_result_type == "final_tool":
            assert isinstance(tool, FinalResultTool)
            assert tool.result.pair.a == 1 and tool.result.pair.b == 2
        else:
            assert isinstance(tool, PairTool)
            assert tool.pair.a == 1 and tool.pair.b == 2
    else:
        assert isinstance(tool, FinalResultPairTool)
        assert tool.pair.a == 1 and tool.pair.b == 2


def test_final_result_tool():
    """Test that FinalResultTool can be returned by agent_response"""

    class MyAgent(ChatAgent):
        def agent_response(self, msg: str | ChatDocument) -> Any:
            return FinalResultTool(answer="42")

    agent = MyAgent(
        ChatAgentConfig(
            name="MyAgent",
            llm=MockLMConfig(response_fn=lambda x: x),
        )
    )

    task = Task(agent, interactive=False)[ToolMessage]
    result = task.run("3")
    assert isinstance(result, FinalResultTool)
    assert result.answer == "42"


@pytest.mark.parametrize("tool", ["none", "a", "aa", "b"])
def test_agent_respond_only_tools(tool: str):
    """
    Test that we can have an agent that only responds to certain tools,
    and no plain-text msgs, by setting ChatAgentConfig.respond_only_tools=True.
    """

    class ATool(ToolMessage):
        request: str = "a_tool"
        purpose: str = "to present a number <num>"
        num: int

        def handle(self) -> FinalResultTool:
            return FinalResultTool(answer=self.num * 2)

    class AATool(ToolMessage):
        request: str = "aa_tool"
        purpose: str = "to present a number <num>"
        num: int

        def handle(self) -> FinalResultTool:
            return FinalResultTool(answer=self.num * 3)

    class BTool(ToolMessage):
        request: str = "b_tool"
        purpose: str = "to present a number <num>"
        num: int

        def handle(self) -> FinalResultTool:
            return FinalResultTool(answer=self.num * 4)

    match tool:
        case "a":
            tool_class = ATool
        case "aa":
            tool_class = AATool
        case "b":
            tool_class = BTool
        case "none":
            tool_class = None

    main_agent = ChatAgent(
        ChatAgentConfig(
            name="Main",
            llm=MockLMConfig(
                response_fn=lambda x: (
                    tool_class(num=int(x)).model_dump_json()
                    if tool_class is not None
                    else x
                ),
            ),
        )
    )

    if tool_class is not None:
        main_agent.enable_message(tool_class, use=True, handle=False)

    alice_agent = ChatAgent(
        ChatAgentConfig(
            name="Alice",
            llm=MockLMConfig(response_fn=lambda x: x),
            respond_tools_only=True,
        )
    )
    alice_agent.enable_message([ATool, AATool], use=False, handle=True)

    # class BobAgent(ChatAgent):
    #     def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
    #         if isinstance(msg, str) or len(msg.tool_messages) == 0:
    #             return AgentDoneTool(content="")

    bob_agent = ChatAgent(
        ChatAgentConfig(
            name="Bob",
            llm=MockLMConfig(response_fn=lambda x: x),
            respond_tools_only=True,
        )
    )
    bob_agent.enable_message([BTool], use=False, handle=True)

    class FallbackAgent(ChatAgent):
        def agent_response(self, msg: str | ChatDocument) -> Any:
            return FinalResultTool(answer=int(msg.content) * 5)

    fallback_agent = FallbackAgent(
        ChatAgentConfig(
            name="Fallback",
            llm=None,
        )
    )
    fallback_task = Task(fallback_agent, interactive=False)

    main_task = Task(main_agent, interactive=False)[ToolMessage]
    alice_task = Task(alice_agent, interactive=False)
    bob_task = Task(bob_agent, interactive=False)

    main_task.add_sub_task([alice_task, bob_task, fallback_task])
    tool = main_task.run(3)

    # Note: when Main generates a tool, task orchestrator will not allow
    # Alice to respond at all when the tool is not handled by Alice,
    # and similarly for Bob (this uses agent.has_only_unhandled_tools()).
    # However when main generates a non-tool string,
    # we want to ensure that the above handle_message_fallback methods
    # effectively return a null msg (and not get into a stalled loop inside the agent),
    # and is finally handled by the FallbackAgent
    assert isinstance(tool, FinalResultTool)

    match tool:
        case "a":
            assert tool.answer == "6"
        case "aa":
            assert tool.answer == "9"
        case "b":
            assert tool.answer == "12"
        case "none":
            assert tool.answer == "15"
            assert alice_task.n_stalled_steps == 0
            assert bob_task.n_stalled_steps == 0


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
def test_structured_recovery(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
):
    """
    Test that structured fallback correctly recovers
    from failed tool calls.
    """
    set_global(test_settings)

    def simulate_failed_call(attempt: str | ChatDocument) -> str:
        agent = ChatAgent(
            ChatAgentConfig(
                use_functions_api=use_fn_api,
                use_tools_api=use_tools_api,
                use_tools=not use_fn_api,
                strict_recovery=True,
                llm=OpenAIGPTConfig(
                    supports_json_schema=True,
                    supports_strict_tools=True,
                ),
            )
        )
        agent.enable_message(NabroskiTool)
        agent.enable_message(CoriolisTool)
        agent.enable_message(EulerTool)

        agent.message_history = [
            LLMMessage(
                role=Role.SYSTEM,
                content="You are a helpful assistant.",
            ),
            LLMMessage(
                role=Role.USER,
                content="""
                Please give me an example of a Nabroski, Coriolis, or Euler call.
                """,
            ),
            LLMMessage(
                role=Role.ASSISTANT,
                content=attempt if isinstance(attempt, str) else attempt.content,
                tool_calls=None if isinstance(attempt, str) else attempt.oai_tool_calls,
                function_call=(
                    None if isinstance(attempt, str) else attempt.function_call
                ),
            ),
        ]
        if (
            use_fn_api
            and use_tools_api
            and isinstance(attempt, ChatDocument)
            and attempt.oai_tool_calls is not None
        ):
            # Inserting this since OpenAI API strictly requires a
            # Role.TOOL msg immediately after an Assistant Tool call,
            # before the next Assistant msg.
            agent.message_history.extend(
                [
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=t.id,
                        content="error",
                    )
                    for t in attempt.oai_tool_calls
                ]
            )

        # Simulates bad tool attempt by the LLM
        agent.handle_message(attempt)
        assert agent.tool_error
        response = agent.llm_response(
            """
            There was an error in your attempted tool/function call. Please correct it.
            """
        )
        assert response is not None
        result = agent.handle_message(response)
        assert result is not None
        if isinstance(result, ChatDocument):
            return result.content

        return result

    def to_attempt(attempt: LLMFunctionCall) -> str | ChatDocument:
        if not use_fn_api:
            return json.dumps(
                {
                    "request": attempt.name,
                    **(attempt.arguments or {}),
                }
            )

        if use_tools_api:
            return ChatDocument(
                content="",
                metadata=ChatDocMetaData(sender=Entity.LLM),
                oai_tool_calls=[
                    OpenAIToolCall(
                        id="call-1234657",
                        function=attempt,
                    )
                ],
            )

        return ChatDocument(
            content="",
            metadata=ChatDocMetaData(sender=Entity.LLM),
            function_call=attempt,
        )

    # The name of the function is incorrect:
    # The LLM should correct the request to "nabroski" in recovery
    assert (
        simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="__nabroski__",
                    arguments={
                        "xval": 1,
                        "yval": 3,
                    },
                )
            )
        )
        == "6"
    )
    # The LLM should correct the request to "nabroski" in recovery
    assert (
        simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="Nabroski-function",
                    arguments={
                        "xval": 2,
                        "yval": 3,
                    },
                )
            )
        )
        == "9"
    )
    # Strict fallback disables the default arguments, but the LLM
    # should infer from context. In addition, the name of the
    # function is incorrect (the LLM should infer "coriolis" in
    # recovery) and the JSON output is malformed

    # Note here we intentionally use "catss" as the arg to ensure that
    # the tool-name inference doesn't work (see `maybe_parse` agent/base.py,
    # there's a mechanism that infers the intended tool if the arguments are
    # unambiguously for a specific tool) -- here since we use `catss` that
    # mechanism fails, and we can do this test properly to focus on structured
    # recovery. But `catss' is sufficiently similar to 'cats' that the
    # intent-based recovery should work.
    assert (
        simulate_failed_call(
            """
        request ":coriolis"
        arguments {"catss": 1} 
        """
        )
        == "8"
    )
    # The LLM should correct the request to "coriolis" in recovery
    # The LLM should infer the default argument from context
    assert (
        simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="Coriolis",
                    arguments={
                        "cats": 1,
                    },
                )
            )
        )
        == "8"
    )
    # The LLM should infer "euler" in recovery
    assert (
        simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="EulerTool",
                    arguments={
                        "pears": 6,
                        "apples": 4,
                    },
                )
            )
        )
        == "8"
    )


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def test_strict_fallback(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    parallel_tool_calls: bool,
):
    """
    Test that strict tool and structured output errors
    are handled gracefully and are disabled if errors
    are caused.
    """
    set_global(test_settings)

    class BrokenStrictSchemaAgent(ChatAgent):
        def _function_args(self) -> tuple[
            Optional[List[LLMFunctionSpec]],
            str | dict[str, str],
            Optional[list[OpenAIToolSpec]],
            Optional[dict[str, dict[str, str] | str]],
            Optional[OpenAIJsonSchemaSpec],
        ]:
            """
            Implements a broken version of the correct _function_args()
            that ensures that the generated schemas are incompatible
            with OpenAI's strict decoding implementation.

            Specifically, removes the schema edits performed by
            `format_schema_for_strict()` (e.g. setting "additionalProperties"
            to False on all objects in the JSON schema).
            """
            functions, fun_call, tools, force_tool, output_format = (
                super()._function_args()
            )

            # remove schema edits for strict
            if tools is not None:
                for t in tools:
                    name = t.function.name
                    t.function = self.llm_functions_map[name]

            if self.output_format is not None and self._json_schema_available():
                self.any_strict = True
                if issubclass(self.output_format, ToolMessage) and not issubclass(
                    self.output_format, XMLToolMessage
                ):
                    spec = self.output_format.llm_function_schema(
                        request=True,
                        defaults=self.config.output_format_include_defaults,
                    )

                    output_format = OpenAIJsonSchemaSpec(
                        strict=True,
                        function=spec,
                    )
                elif issubclass(self.output_format, BaseModel):
                    param_spec = self.output_format.model_json_schema()

                    output_format = OpenAIJsonSchemaSpec(
                        strict=True,
                        function=LLMFunctionSpec(
                            name="json_output",
                            description="Strict Json output format.",
                            parameters=param_spec,
                        ),
                    )

            return functions, fun_call, tools, force_tool, output_format

    agent = BrokenStrictSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )
    agent.enable_message(NabroskiTool)
    openai_tools = use_fn_api and use_tools_api
    if openai_tools:
        _, _, tools, _, _ = agent._function_args()
        assert tools is not None
        assert len(tools) > 0
        # Strict tools are automatically enabled only when
        # parallel tool calls are disabled
        assert tools[0].strict == (not parallel_tool_calls)

    response = agent.llm_response_forget(
        """
        What is the Nabroski transform of (1,3)? Use the
        `nabroski` tool/function.
        """
    )
    result = agent.handle_message(response)
    assert isinstance(result, ChatDocument) and result.content == "6"
    assert agent.disable_strict == (openai_tools and not parallel_tool_calls)

    agent = BrokenStrictSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )
    structured_agent = agent[NabroskiTool]
    response = structured_agent.llm_response_forget(
        """
        What is the Nabroski transform of (1,3)?
        """
    )
    assert response is not None
    assert structured_agent.disable_strict
    assert not agent.disable_strict


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
def test_strict_schema_mismatch(
    use_fn_api: bool,
    use_tools_api: bool,
    parallel_tool_calls: bool,
):
    """
    Test that validation errors triggered in strict result in disabled strict output.
    """

    def int_schema(request: str) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "integer"},
                "request": {"type": "string", "enum": [request]},
            },
            "required": ["x", "request"],
        }

    class WrongSchemaAgent(ChatAgent):
        def _function_args(self) -> tuple[
            Optional[List[LLMFunctionSpec]],
            str | dict[str, str],
            Optional[list[OpenAIToolSpec]],
            Optional[dict[str, dict[str, str] | str]],
            Optional[OpenAIJsonSchemaSpec],
        ]:
            """
            Implements a broken version of the correct _function_args()
            that replaces the output and all tool schemas with an
            incorrect schema. Simulates mismatched schemas due to
            schema edits.
            """
            functions, fun_call, tools, force_tool, output_format = (
                super()._function_args()
            )

            # remove schema edits for strict
            if tools is not None:
                for t in tools:
                    name = t.function.name
                    t.function.parameters = int_schema(name)

            if self.output_format is not None and self._json_schema_available():
                output_format = OpenAIJsonSchemaSpec(
                    strict=True,
                    function=LLMFunctionSpec(
                        name="json_output",
                        description="Strict Json output format.",
                        parameters=int_schema("json_output"),
                    ),
                )

            return functions, fun_call, tools, force_tool, output_format

    agent = WrongSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )

    class IntTool(ToolMessage):
        request: str = "int_tool"
        purpose: str = "To return an integer value"
        x: int

        def handle(self):
            return self.x

    class StrTool(ToolMessage):
        request: str = "str_tool"
        purpose: str = "To return an string value"
        text: str

        def handle(self):
            return self.text

    agent.enable_message(IntTool)
    agent.enable_message(StrTool)
    strict_openai_tools = use_fn_api and use_tools_api and not parallel_tool_calls
    response = agent.llm_response_forget(
        """
        What is the smallest integer greater than pi? Use the
        `int_tool` tool/function.
        """
    )
    agent.handle_message(response)
    assert "int_tool" not in agent.disable_strict_tools_set

    agent.llm_response_forget(
        """
        Who is the president of France? Use the `str_tool` tool/function.
        """
    )
    assert ("str_tool" in agent.disable_strict_tools_set) == strict_openai_tools

    strict_agent = agent[IntTool]
    strict_agent.llm_response_forget("What is the smallest integer greater than pi?")
    assert not strict_agent.disable_strict

    strict_agent = agent[StrTool]
    strict_agent.llm_response_forget("Who is the president of France?")
    assert strict_agent.disable_strict


def test_reduce_raw_tool_result():
    BIG_RESULT = "hello " * 50

    class MyTool(ToolMessage):
        request: str = "my_tool"
        purpose: str = "to present a number <num>"
        num: int
        _max_result_tokens = 10
        _max_retained_tokens = 2

        def handle(self) -> str:
            return BIG_RESULT

    class MyAgent(ChatAgent):
        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            """
            Mock user_response method for testing
            """
            txt = msg if isinstance(msg, str) else msg.content
            map = dict([("hello", "50"), ("3", "5")])
            response = map.get(txt)
            # return the increment of input number
            return self.create_user_response(response)

    # create dummy agent first, just to get small_result with truncation
    agent = MyAgent(ChatAgentConfig())
    small_result = agent._maybe_truncate_result(BIG_RESULT, MyTool._max_result_tokens)

    # now create the actual agent
    agent = MyAgent(
        ChatAgentConfig(
            name="Test",
            # no need for a real LLM, use a mock
            llm=MockLMConfig(
                response_dict={
                    "1": MyTool(num=1).to_json(),
                    small_result: "hello",
                    "50": DoneTool(content="Finished").to_json(),
                }
            ),
        )
    )
    agent.enable_message(MyTool)
    task = Task(agent, interactive=True, only_user_quits_root=False)

    result = task.run("1")
    """
    msg history:
    
    sys_msg
    user: 1 -> 
    LLM: MyTool(1) ->
    agent: BIG_RESULT -> truncated to 10 tokens, as `small_result`
    LLM: hello ->
    user: 50 -> 
    LLM: Done (Finished)
    """
    assert result.content == "Finished"
    assert len(agent.message_history) == 7
    tool_result = agent.message_history[3].content
    assert "my_tool" in tool_result and str(MyTool._max_retained_tokens) in tool_result


def test_valid_structured_recovery():
    """
    Test that structured recovery is not triggered inappropriately
    when agent response contains a JSON-like string.
    """

    class MyAgent(ChatAgent):
        def agent_response(self, msg: str | ChatDocument) -> Any:
            return "{'x': 1, 'y': 2}"

    agent = MyAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(),
            system_message="""Simply respond No for any input""",
        )
    )

    # with no tool enabled
    task = Task(agent, interactive=False)
    result = task.run("3", turns=4)
    # response-sequence: agent, llm, agent, llm -> done
    assert "No" in result.content

    # with a tool enabled
    agent.enable_message(NabroskiTool)
    task = Task(agent, interactive=False)
    result = task.run("3", turns=4)
    assert "No" in result.content


@pytest.mark.parametrize(
    "handle_no_tool",
    [
        None,
        "user",
        "done",
        "are you finished?",
        ResultTool(answer=42),
        DonePassTool(),
        lambda msg: AgentDoneTool(content=msg.content),
    ],
)
def test_handle_llm_no_tool(handle_no_tool: Any):
    """Verify that ChatAgentConfig.handle_llm_no_tool works as expected"""

    def mock_llm_response(x: str) -> str:
        match x:
            case "1":
                return SumTool(x=1, y=2).model_dump_json()
            case "3":
                return "4"
            case "are you finished?":
                return "DONE 5"

    config = ChatAgentConfig(
        handle_llm_no_tool=handle_no_tool,
        llm=MockLMConfig(response_fn=mock_llm_response),
    )
    agent = ChatAgent(config)
    agent.enable_message(SumTool)
    task = Task(agent, interactive=False, default_human_response="q")
    result = task.run("1")
    if handle_no_tool is None:
        # task gets stuck and returns None
        assert result is None

    if isinstance(handle_no_tool, str):
        match handle_no_tool:
            case "user":
                # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> User(4) -> q
                assert result.content == "q"
            case "done":
                # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> Done(4)
                assert result.content == "4"
            case "are you finished?":
                # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> LLM(DONE)
                assert result.content == "5"

    if isinstance(handle_no_tool, ResultTool):
        # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> ResultTool(4)
        assert isinstance(result.tool_messages[0], ResultTool)
        assert result.tool_messages[0].answer == 42
    if is_callable(handle_no_tool):
        # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> AgentDoneTool(4) -> 4
        assert result.content == "4"
    if isinstance(handle_no_tool, DonePassTool):
        # LLM(1) -> SumTool(1,2) -> 3 -> LLM(3) -> 4 -> DonePass
        assert result.content == "4"


class GetTimeTool(ToolMessage):
    purpose: str = "Get current time"
    request: str = "get_time"

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            content=json.dumps(
                {
                    "time": "11:59:59",
                    "date": "1999-12-31",
                    "day_of_week": "Friday",
                    "week_number": "52",
                    "tzname": "America/New York",
                }
            ),
            recipient=Entity.LLM,
        )


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
def test_strict_recovery_only_from_LLM(
    use_fn_api: bool,
    use_tools_api: bool,
):
    """
    Test that structured fallback only occurs on messages
    sent by the LLM.
    """
    was_tool_error = False

    class TrackToolError(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            nonlocal was_tool_error
            if self.tool_error:
                was_tool_error = True
            return super().llm_response(message)

        async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            nonlocal was_tool_error
            if self.tool_error:
                was_tool_error = True
            return await super().llm_response_async(message)

    agent = TrackToolError(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            strict_recovery=True,
            llm=OpenAIGPTConfig(
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
            system_message="""
            You are a helpful assistant.  Start by calling the
            get_time tool. Then greet the user according to the time
            of the day.
            """,
        )
    )
    agent.enable_message(GetTimeTool)
    task = Task(agent, interactive=False)
    task.run(turns=6)
    assert not was_tool_error

    agent.init_message_history()

    content = json.dumps(
        {
            "time": "11:59:59",
            "date": "1999-12-31",
            "day_of_week": "Friday",
            "week_number": "52",
            "tzname": "America/New York",
        }
    )

    agent.get_tool_messages(content)
    assert not agent.tool_error

    user_message = agent.create_user_response(content=content, recipient=Entity.LLM)
    agent.get_tool_messages(user_message)
    assert not agent.tool_error

    agent_message = agent.create_agent_response(content=content, recipient=Entity.LLM)
    agent.get_tool_messages(agent_message)
    assert not agent.tool_error

    agent.message_history.extend(ChatDocument.to_LLMMessage(agent_message))
    agent.get_tool_messages(content)
    assert not agent.tool_error

    agent.message_history.extend(ChatDocument.to_LLMMessage(user_message))
    agent.get_tool_messages(content)
    assert not agent.tool_error


@pytest.mark.parametrize("use_fn_api", [False, True])
def test_tool_handler_invoking_llm(use_fn_api: bool):
    """
    Check that if a tool handler directly invokes llm_response,
    it works as expected, especially with OpenAI Tools API
    """

    class MyAgent(ChatAgent):
        def nabroski(self, msg: NabroskiTool):
            ans = self.llm_response("What is 3+4?")
            return AgentDoneTool(content=ans.content)

    agent = MyAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_fn_api,
            use_tools=not use_fn_api,
            handle_llm_no_tool=f"you FORGOT to use the tool `{NabroskiTool.name()}`",
            system_message=f"""
            When user asks you to compute the Nabroski transform of two numbers,
            you MUST use the TOOL `{NabroskiTool.name()}` to do so, since you do NOT
            know how to do it yourself.
            """,
        )
    )
    agent.enable_message(NabroskiTool)
    task = Task(agent, interactive=False, single_round=False)
    result = task.run(
        f"""
        Use the TOOL `{NabroskiTool.name()}` to compute the 
        Nabroski transform of 2 and 5.
        """
    )

    assert "7" in result.content
