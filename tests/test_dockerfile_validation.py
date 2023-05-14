# from examples.dockerchat.docker_chat_agent import DockerChatAgent

# import os
# import tempfile
# import subprocess

import pytest

# from unittest.mock import patch, MagicMock

from llmagent.parsing.json import extract_top_level_json
from llmagent.utils.configuration import update_global_settings
from llmagent.language_models.base import Role, LLMMessage
from llmagent.agent.base import AgentConfig, Agent
from llmagent.language_models.base import LLMConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.agent.base import AgentMessage
from llmagent.agent.chat_agent import ChatAgent
from llmagent.utils.system import rmdir

from typing import List
from functools import reduce

import json
import re


class ValidateDockerfileMessage(AgentMessage):
    request: str = "validate_dockerfile"
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
            "Here is a sample Dockerfile",
            "You can modify this Dockerfile",
            "Does this look good to you",
            "Here is the Dockerfile",
            "This Dockerfile installs",
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
    llm=LLMConfig(
        type="openai",
        chat_model="gpt-3.5-turbo",
    ),
    parsing=None,
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


@pytest.mark.parametrize("msg_cls", [ValidateDockerfileMessage])
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls().usage_example()
    assert any(
        template in usage
        for template in reduce(
            lambda x, y: x + y, [ex.use_when() for ex in msg_cls.examples()]
        )
    )


rmdir(qd_dir)  # don't need it here

df = "FROM ubuntu:latest\nLABEL maintainer=blah"

df_json = json.dumps(df)

NONE_MSG = "nothing to see here"

VALIDATE_DOCKERFILE_MSG = f"""
Ok, thank you.
{{
'request': 'validate_dockerfile',
'proposed_dockerfile': {df_json}
}} 
this is the dockerfile
""" 

FILE_EXISTS_MSG = """
Ok, thank you.
{
'request': 'file_exists',
'filename': 'test.txt'
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
    pieces = [s.replace("\\n", "") for s in  string.split()]
    return "".join(pieces)

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
    agent.enable_message(ValidateDockerfileMessage)

    df = """
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
    """
    # need to conert to proper json format, else can cause json parse error
    df_json = json.dumps(df)

    msg = (
        """
    here is the dockerfile
    {"request": "validate_dockerfile",
    "proposed_dockerfile": "%s"}
    """
        % df_json
    )

    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    ld_reformatted_json = json.loads(reformatted_jsons[0])
    proposed_dockerfile_nowhitespace = clean_string(
        ld_reformatted_json.get("proposed_dockerfile")
    )

    ld_reformatted_json["proposed_dockerfile"] = proposed_dockerfile_nowhitespace
    df_nowhitespace = clean_string(df)
 
    assert len(reformatted_jsons) == 1
    assert ld_reformatted_json == ValidateDockerfileMessage(
        proposed_dockerfile=f"{df_nowhitespace}"
    ).dict(exclude={"result"})

