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

df = """
                FROM ubuntu:latest
                LABEL maintainer=blah
                """


@pytest.mark.parametrize(
    "message, expected",
    [
        ("nothing to see here", None),
        (
            """Ok, thank you. 
                {
                'request': 'validate_dockerfile',
                'proposed_dockerfile':'doeckfile definition'
                } 
                this is the Dockerfile!
                """,
            "Built successfully",
        ),
    ],
)
def test_agent_actions(message, expected):
    """
    Test whether messages are handled correctly.
    """
    agent.enable_message(ValidateDockerfileMessage)
    result = agent.handle_message(message)
    assert result == expected


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
    msg = (
        """
    here is the dockerfile
    {"request": "validate_dockerfile",
    "proposed_dockerfile": "%s"}
    """
        % df
    )

    prompt = agent.request_reformat_prompt(msg)
    reformat_agent = Agent(cfg)
    reformatted = reformat_agent.respond(prompt)
    reformatted_jsons = extract_top_level_json(reformatted.content)
    assert len(reformatted_jsons) == 1
    assert (
        json.loads(reformatted_jsons[0])
        == ValidateDockerfileMessage(
            proposed_dockerfile="""
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
        """
        ).dict(exclude={"result"})
    )
