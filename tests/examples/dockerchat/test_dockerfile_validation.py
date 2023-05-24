from llmagent.utils.configuration import update_global_settings, Settings, set_global
from llmagent.language_models.base import Role, LLMMessage
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.agent.base import AgentConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.agent.base import AgentMessage
from llmagent.agent.chat_agent import ChatAgent
from llmagent.utils.system import rmdir
from llmagent.cachedb.redis_cachedb import RedisCacheConfig

from typing import List


import json


class ValidateDockerfileMessage(AgentMessage):
    request: str = "validate_dockerfile"
    purpose: str = "To check whether a Dockerfile is valid."
    proposed_dockerfile: str = """
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
        """  # contents of dockerfile
    result: str = "build succeed"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(
                proposed_dockerfile="""
                FROM ubuntu:latest
                LABEL maintainer=blah
                """,
                result="received, but there are errors",
            ),
            cls(
                proposed_dockerfile="""
                # Use an official Python runtime as a parent image
                FROM python:3.7-slim
                # Set the working directory in the container
                WORKDIR /app
                """,
                result="docker file looks fine",
            ),
        ]


class MessageHandlingAgent(ChatAgent):
    def validate_dockerfile(self, ValidateDockerfileMessage) -> str:
        return "Built successfully"


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = AgentConfig(
    debug=True,
    name="test-llmagent",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


def test_enable_message():
    agent.enable_message(ValidateDockerfileMessage)
    assert "validate_dockerfile" in agent.handled_classes
    assert agent.handled_classes["validate_dockerfile"] == ValidateDockerfileMessage


def test_disable_message():
    agent.enable_message(ValidateDockerfileMessage)
    agent.disable_message(ValidateDockerfileMessage)
    assert "validate_dockerfile" not in agent.handled_classes


rmdir(qd_dir)  # don't need it here

df = "FROM ubuntu:latest\nLABEL maintainer=blah"

df_json = json.dumps(df)

NONE_MSG = "nothing to see here"

VALIDATE_DOCKERFILE_MSG = f"""
Ok, thank you.
{{
"request": "validate_dockerfile",
"proposed_dockerfile": {df_json}
}} 
this is the dockerfile
"""

FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists",
"filename": "test.txt"
} 
Hope you can tell me!
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(ValidateDockerfileMessage)
    assert agent.handle_message(NONE_MSG) is None
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) == "Built successfully"

    agent.disable_message(ValidateDockerfileMessage)
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) is None

    agent.enable_message(ValidateDockerfileMessage)
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) == "Built successfully"


def test_llm_agent_message(test_settings: Settings):
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    update_global_settings(cfg, keys=["debug"])
    set_global(test_settings)
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
    agent.enable_message(ValidateDockerfileMessage)

    agent.run(
        iters=2, default_human_response="I don't know, please ask your next question."
    )


def clean_string(string: str) -> str:
    """
    removes whitespace in possibly mutli-line string
    Args:
        s(str): string to be modified
    Returns:
        string after cleaning up whitespace
    """
    pieces = [s.replace("\\n", "") for s in string.split()]
    return "".join(pieces)
