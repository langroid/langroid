from llmagent.agent.base import AgentConfig
from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.message import AgentMessage
from llmagent.language_models.openai_gpt import (
    OpenAIGPTConfig,
    OpenAIChatModel,
)
from llmagent.parsing.json import extract_top_level_json
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.system import rmdir
from llmagent.utils.configuration import update_global_settings, Settings, set_global
from typing import List
import pytest
import json


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
    filename: str = "test.txt"
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
cfg = AgentConfig(
    debug=True,
    name="test-llmagent",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


def test_enable_message():
    agent.enable_message(FileExistsMessage)
    assert "file_exists" in agent.handled_classes
    assert agent.handled_classes["file_exists"] == FileExistsMessage

    agent.enable_message(PythonVersionMessage)
    assert "python_version" in agent.handled_classes
    assert agent.handled_classes["python_version"] == PythonVersionMessage


def test_disable_message():
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

    agent.disable_message(FileExistsMessage)
    assert "file_exists" not in agent.handled_classes

    agent.disable_message(PythonVersionMessage)
    assert "python_version" not in agent.handled_classes


@pytest.mark.parametrize("msg_cls", [PythonVersionMessage, FileExistsMessage])
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls().usage_example()
    assert json.loads(usage)["request"] == msg_cls().request


rmdir(qd_dir)  # don't need it here

NONE_MSG = "nothing to see here"

FILE_EXISTS_MSG = """
Ok, thank you.
{
'request': 'file_exists',
'filename': 'test.txt'
} 
Hope you can tell me!
"""

PYTHON_VERSION_MSG = """
great, please tell me this --
{
'request': 'python_version'
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

    agent.disable_message(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"

    agent.disable_message(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"


def test_llm_agent_message(test_settings: Settings):
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    set_global(test_settings)
    update_global_settings(cfg, keys=["debug"])
    agent = MessageHandlingAgent(cfg)
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(CountryCapitalMessage)

    llm_msg = agent.respond("Start by asking me about the python version.").content

    agent_result = agent.handle_message(llm_msg)
    assert agent_result == "3.9"

    agent.clear_history(-2)
    llm_msg = agent.respond(
        "Start by asking me whether file 'requirements.txt' exists."
    ).content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result == "yes"

    agent.clear_history(-2)
    llm_msg = agent.respond(
        "Start by asking me whether Paris is the capital of France."
    ).content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result == "yes"

    agent.clear_history(-2)
    llm_msg = agent.respond("Ask me to check what is the population of France.").content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result is None


def test_llm_agent_reformat(test_settings: Settings):
    """
    Test whether the LLM completion mode is able to reformat the request based
    on the auto-generated reformat instructions.
    """
    update_global_settings(cfg, keys=["debug"])
    set_global(test_settings)

    agent = MessageHandlingAgent(cfg)
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(CountryCapitalMessage)

    FILE = "blah.txt"
    reformatted = agent.reformat_message(
        f"I want to know if the repo contains the file '{FILE}'"
    )
    reformatted_jsons = extract_top_level_json(reformatted)
    assert len(reformatted_jsons) == 1
    assert (
        json.loads(reformatted_jsons[0])
        == FileExistsMessage(filename=FILE).dict_example()
    )

    reformatted = agent.reformat_message(
        "I want to know which version of Python is needed"
    )
    reformatted_jsons = extract_top_level_json(reformatted)
    assert len(reformatted_jsons) == 1
    assert json.loads(reformatted_jsons[0]) == PythonVersionMessage().dict_example()

    reformatted = agent.reformat_message("I need to know the population of England")
    reformatted_jsons = extract_top_level_json(reformatted)
    assert len(reformatted_jsons) == 0

    COUNTRY = "India"
    CITY = "Delhi"
    reformatted = agent.reformat_message(
        f"Check whether the capital of {COUNTRY} is {CITY}",
    )
    reformatted_jsons = extract_top_level_json(reformatted)
    assert len(reformatted_jsons) == 1
    assert (
        json.loads(reformatted_jsons[0])
        == CountryCapitalMessage(
            country=COUNTRY,
            city=CITY,
        ).dict_example()
    )
