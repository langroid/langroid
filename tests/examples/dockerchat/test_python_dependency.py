import os
import tempfile
import pytest
import json


from llmagent.parsing.json import extract_top_level_json
from llmagent.utils.configuration import update_global_settings
from llmagent.language_models.base import Role, LLMMessage
from llmagent.agent.base import AgentConfig, Agent
from llmagent.language_models.base import LLMConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.agent.base import AgentMessage
from llmagent.agent.chat_agent import ChatAgent
from llmagent.utils.system import rmdir
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from examples.dockerchat.identify_python_dependency import (
    identify_dependency_management,
    DEPENDENCY_FILES,
)

from functools import reduce
from typing import List


class PythonDependencyMessage(AgentMessage):
    request: str = "python_dependency"
    result: str = "yes"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(result="This repo uses requirements.txt for managing dependencies"),
            cls(result="This repo uses pyproject.toml for managing dependencies"),
            cls(result="This repo doesn't contain any dependacy manager"),
        ]

    def use_when(self) -> List[str]:
        """
        Return a List of strings showing an example of when the message should be used,
        possibly parameterized by the field values. This should be a valid english
        phrase in first person, in the form of a phrase that can legitimately
        complete "I can use this message when..."
        Returns:
            str: list of examples of a situation when the message should be used,
                in first person, possibly parameterized by the field values.
        """

        return [
            "what are the dependencies in the repo.",
            "I need to check if the repo contains dependencies",
        ]


class MessageHandlingAgent(ChatAgent):
    def python_dependency(self, PythonDependencyMessage) -> str:
        return "This repo uses requirements.txt for managing dependencies"


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = AgentConfig(
    debug=True,
    name="test-llmagent",
    vecdb=None,
    llm=LLMConfig(
        type="openai",
        chat_model="gpt-3.5-turbo",
        cache_config=RedisCacheConfig(fake=True),
    ),
    parsing=None,
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


def test_enable_message():
    agent.enable_message(PythonDependencyMessage)
    assert "python_dependency" in agent.handled_classes
    assert agent.handled_classes["python_dependency"] == PythonDependencyMessage


def test_disable_message():
    agent.enable_message(PythonDependencyMessage)
    agent.disable_message(PythonDependencyMessage)
    assert "python_dependency" not in agent.handled_classes


@pytest.mark.parametrize("msg_cls", [PythonDependencyMessage])
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls().usage_example()
    assert any(
        template in usage
        for template in reduce(
            lambda x, y: x + y, [ex.use_when() for ex in msg_cls.examples()]
        )
    )


rmdir(qd_dir)  # don't need it here


NONE_MSG = "nothing to see here"

PYTHON_DEPENDENCY_MSG = """
great, please tell me this --
{
'request': 'python_dependency'
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

    agent.disable_message(PythonDependencyMessage)
    assert agent.handle_message(PYTHON_DEPENDENCY_MSG) is None

    agent.enable_message(PythonDependencyMessage)
    assert (
        agent.handle_message(PYTHON_DEPENDENCY_MSG)
        == "This repo uses requirements.txt for managing dependencies"
    )


def test_llm_agent_message():
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    update_global_settings(cfg, keys=["debug"])
    task_messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""
            You are a devops engineer, and your task is to understand a PYTHON 
            repo. Plan this out step by step, and ask me questions 
            for any info you need to understand the repo. 
            """,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
            You are an assistant whose task is to understand a Python repo.

            You have to think in small steps, and at each stage, show me your 
            THINKING, and the QUESTION you want to ask. Based on my answer, you will 
            generate a new THINKING and QUESTION.  
            """,
        ),
    ]
    agent = MessageHandlingAgent(cfg, task_messages)
    agent.enable_message(PythonDependencyMessage)

    agent.run(
        iters=2, default_human_response="I don't know, please ask your next question."
    )


def test_llm_agent_reformat():
    """
    Test whether the LLM completion mode is able to reformat the request based
    on the auto-generated reformat instructions.
    """
    update_global_settings(cfg, keys=["debug"])
    task_messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""
            You are a devops engineer, and your task is to understand a PYTHON 
            repo. Plan this out step by step, and ask me questions 
            for any info you need to understand the repo. 
            """,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
            You are an assistant whose task is to understand a Python repo.

            You have to think in small steps, and at each stage, show me your 
            THINKING, and the QUESTION you want to ask. Based on my answer, you will 
            generate a new THINKING and QUESTION.  
            """,
        ),
    ]
    agent = MessageHandlingAgent(cfg, task_messages)
    agent.enable_message(PythonDependencyMessage)

    msg = """
    I want to know the dependencies in this project
    """

    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    assert len(reformatted_jsons) == 1
    assert json.loads(reformatted_jsons[0]) == PythonDependencyMessage().dict(
        exclude={"result"}
    )


@pytest.mark.parametrize("depfile", DEPENDENCY_FILES)
def test_identify_dependency_management(depfile):
    # Test case 1: Check for requirements.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, depfile), "w") as f:
            f.write("")
        found_deps = identify_dependency_management(tmpdir)
        assert found_deps == [] if depfile == "junk.txt" else [depfile]
