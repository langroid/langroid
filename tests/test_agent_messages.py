from llmagent.agent.base import AgentConfig, Agent
from llmagent.language_models.base import Role, LLMMessage
from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.message import AgentMessage, ThoughtQuestionAnswer
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.json import extract_top_level_json
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.system import rmdir
from llmagent.utils.configuration import update_global_settings
from typing import List
from functools import reduce
import pytest
import json


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"
    filename: str = "test.txt"
    result: str = "yes"  # or "no"

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md", result="yes"),
            cls(filename="Dockerfile", result="no"),
        ]

    def use_when(self):
        return [
            f"I want to know if a file {self.filename} is in the repo",
            f"I need to check if the repo contains the file {self.filename}",
        ]

    def non_usage_examples(self) -> List[ThoughtQuestionAnswer]:
        return [
            ThoughtQuestionAnswer(
                thought="I want to see how many files are in the repo",
                question="How many files are in the repo?",
                answer="34",
            ),
            ThoughtQuestionAnswer(
                thought="I need to know the URL of the repo",
                question="What is the URL of the repo?",
                answer="https://a.github.com/b/c",
            ),
        ]


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"
    result: str = "3.9"

    @classmethod
    def examples(cls) -> List["PythonVersionMessage"]:
        return [
            cls(result="3.7"),
            cls(result="3.8"),
        ]

    def use_when(self):
        return [
            "I want to know which version of Python is needed",
            "I need to check the Python version",
        ]


class MessageHandlingAgent(ChatAgent):
    def file_exists(self, message: FileExistsMessage) -> str:
        return "yes" if message.filename == "requirements.txt" else "no"

    def python_version(self, PythonVersionMessage) -> str:
        return "3.9"


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
    assert any(
        template in usage
        for template in reduce(
            lambda x, y: x + y, [ex.use_when() for ex in msg_cls.examples()]
        )
    )


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
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

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
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

    msg = """
    I want to know whether the repo contains the file 'requirements.txt'
    """

    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    assert len(reformatted_jsons) == 1
    assert json.loads(reformatted_jsons[0]) == FileExistsMessage(
        filename="requirements.txt"
    ).dict(exclude={"result"})

    msg = """
    I want to know which version of Python is needed
    """
    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    assert len(reformatted_jsons) == 1
    assert json.loads(reformatted_jsons[0]) == PythonVersionMessage().dict(
        exclude={"result"}
    )

    msg = """
    I need to know the population of France
    """
    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    assert len(reformatted_jsons) == 0
