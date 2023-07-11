import json
import os
import tempfile
from typing import List

import pytest

from examples_dev.dockerchat.identify_python_dependency import (
    DEPENDENCY_FILES,
    identify_dependency_management,
)
from langroid.agent.base import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.system import rmdir


class PythonDependencyMessage(ToolMessage):
    request: str = "python_dependency"
    purpose: str = "To find out the python dependencies."
    result: str = "yes"

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[ToolMessage]: list of example messages of this type
        """
        return [
            cls(result="This repo uses requirements.txt for managing dependencies"),
            cls(result="This repo uses pyproject.toml for managing dependencies"),
            cls(result="This repo doesn't contain any dependacy manager"),
        ]


class MessageHandlingAgent(ChatAgent):
    def python_dependency(self, PythonDependencyMessage) -> str:
        return "This repo uses requirements.txt for managing dependencies"


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = ChatAgentConfig(
    name="test-langroid",
    system_message="""
    You are a devops engineer, and your task is to understand a PYTHON
    repo. Plan this out step by step, and ask me questions
    for any info you need to understand the repo.
    """,
    user_message="""
    You are an assistant whose task is to understand a Python repo.

    You have to think in small steps, and at each stage, show me your 
    THINKING, and the QUESTION you want to ask. Based on my answer, you will 
    generate a new THINKING and QUESTION.  
    """,
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


@pytest.mark.parametrize("msg_cls", [PythonDependencyMessage])
def test_usage_instruction(msg_cls: ToolMessage):
    usage = msg_cls().usage_example()
    assert json.loads(usage)["request"] == msg_cls().request


rmdir(qd_dir)  # don't need it here


NONE_MSG = "nothing to see here"

PYTHON_DEPENDENCY_MSG = """
great, please tell me this --
{
"request": "python_dependency"
}/if you know it
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(PythonDependencyMessage)
    assert agent.handle_message(NONE_MSG) is None
    assert (
        agent.handle_message(PYTHON_DEPENDENCY_MSG)
        == "This repo uses requirements.txt for managing dependencies"
    )

    agent.disable_message_handling(PythonDependencyMessage)
    assert agent.handle_message(PYTHON_DEPENDENCY_MSG) is None

    agent.enable_message(PythonDependencyMessage)
    assert (
        agent.handle_message(PYTHON_DEPENDENCY_MSG)
        == "This repo uses requirements.txt for managing dependencies"
    )


@pytest.mark.parametrize("depfile", DEPENDENCY_FILES)
def test_identify_dependency_management(depfile):
    # Test case 1: Check for requirements.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, depfile), "w") as f:
            f.write("")
        found_deps = identify_dependency_management(tmpdir)
        assert found_deps == [] if depfile == "junk.txt" else [depfile]
