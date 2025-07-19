from typing import List

import pytest
from pydantic import Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.azure_openai import AzureConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig


class CountryCapitalMessage(ToolMessage):
    request: str = "country_capital"
    purpose: str = "To check whether <city> is the capital of <country>."
    country: str = "France"
    city: str = "Paris"

    @classmethod
    def examples(cls) -> List["CountryCapitalMessage"]:
        return [
            cls(country="France", city="Paris"),
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
    llm=AzureConfig(
        type="azure",
        cache_config=RedisCacheConfig(fake=False),
        deployment_name="langroid-azure-gpt-4o",
        model_name="gpt-4o",
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
)
agent = MessageHandlingAgent(cfg)


@pytest.mark.parametrize("use_functions_api", [False, True])
@pytest.mark.parametrize(
    "message_class, prompt, result",
    [
        (
            FileExistsMessage,
            f"""
            Use the TOOL `{FileExistsMessage.name()}` 
            to check whether the `requirements.txt` exists.
            """,
            "yes",
        ),
        (
            PythonVersionMessage,
            f"""
            Use the TOOL `{PythonVersionMessage.name()}` to 
            check the Python version.
            """,
            "3.9",
        ),
        (
            CountryCapitalMessage,
            f"""
            Use the TOOL `{CountryCapitalMessage.name()}` to check 
            whether the capital of France is Paris.
            """,
            "yes",
        ),
    ],
)
def test_llm_tool_message(
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
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api

    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(CountryCapitalMessage)

    llm_msg = agent.llm_response_forget(prompt)
    tool_name = message_class.default_value("request")
    tools = agent.get_tool_messages(llm_msg)
    assert tools[0].name() == tool_name
    assert len(tools) == 1
    assert isinstance(tools[0], message_class)

    agent_result = agent.handle_message(llm_msg)
    assert result.lower() in agent_result.content.lower()
