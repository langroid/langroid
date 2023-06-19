import itertools
import json
from typing import List, Optional

import pytest
from pydantic import Field

from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.message import AgentMessage
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.configuration import Settings, set_global
from llmagent.utils.system import rmdir


class CountryCapitalMessage(AgentMessage):
    request: str = "country_capital"
    purpose: str = "To check whether <city> is the capital of <country>."
    country: str = "France"
    city: str = "Paris"
    result: str = "yes"  # or "no"

    @classmethod
    def examples(cls) -> List["CountryCapitalMessage"]:
        return [
            cls(country="France", city="Paris", result="yes"),
            cls(country="France", city="Marseille", result="no"),
        ]


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"
    purpose: str = "To check whether a certain <filename> is in the repo."
    filename: str = Field(..., description="File name to check existence of")
    result: str = "yes"  # or "no"

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md", result="yes"),
            cls(filename="Dockerfile", result="no"),
        ]


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"
    purpose: str = "To check which version of Python is needed."
    result: str = "3.9"

    @classmethod
    def examples(cls) -> List["PythonVersionMessage"]:
        return [
            cls(result="3.7"),
            cls(result="3.8"),
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


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = ChatAgentConfig(
    name="test-llmagent",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_llmagent_tools=True,
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
    msg_class: Optional[AgentMessage], use: bool, handle: bool, force: bool
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
def test_disable_message_handling(msg_class: Optional[AgentMessage]):
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
def test_disable_message_use(msg_class: Optional[AgentMessage]):
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
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls.usage_example()
    assert json.loads(usage)["request"] == msg_cls.default_value("request")


rmdir(qd_dir)  # don't need it here

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


@pytest.mark.parametrize(
    "use_functions_api, message_class, prompt, result",
    [
        (
            False,
            FileExistsMessage,
            "Start by asking me whether the file 'requirements.txt' exists",
            "yes",
        ),
        (
            False,
            PythonVersionMessage,
            "Start by asking me about the python version",
            "3.9",
        ),
        (
            False,
            CountryCapitalMessage,
            "Start by asking me whether the capital of France is Paris",
            "yes",
        ),
        (
            True,
            FileExistsMessage,
            "Start by asking me whether the file 'requirements.txt' exists",
            "yes",
        ),
        (
            True,
            PythonVersionMessage,
            "Start by asking me about the python version",
            "3.9",
        ),
        (
            True,
            CountryCapitalMessage,
            "Start by asking me whether the capital of France is Paris",
            "yes",
        ),
    ],
)
def test_llm_agent_message(
    test_settings: Settings,
    use_functions_api: bool,
    message_class: AgentMessage,
    prompt: str,
    result: str,
):
    """
    Test whether LLM is able to GENERATE message (tool) in required format, and the
    agent handles the message correctly.
    Args:
        test_settings: test settings from conftest.py
        use_functions_api: whether to use LLM's functions api or not
            (i.e. use the llmagent AgentMessage tools instead).
        message_class: the message class (i.e. tool/function) to test
        prompt: the prompt to use to induce the LLM to use the tool
        result: the expected result from agent handling the tool-message
    """
    set_global(test_settings)
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_llmagent_tools = not use_functions_api
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


def test_llm_non_tool():
    llm_msg = agent.llm_response_forget(
        "Ask me to check what is the population of France."
    ).content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result is None
