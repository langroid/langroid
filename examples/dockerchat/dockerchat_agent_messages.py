from llmagent.agent.base import AgentMessage
from typing import List
import logging

logger = logging.getLogger(__name__)

class FileExistsMessage(AgentMessage):
    request: str = "file_exists"  # name should exactly match method name in agent
    # below will be fields that will be used by the agent method to handle the message.
    filename: str = "test.txt"
    result: str = "yes"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(filename="requirements.txt", result="yes"),
            cls(filename="test.txt", result="yes"),
            cls(filename="Readme.md", result="no"),
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
            f"I want to know if there is a file named '{self.filename}' in the repo.",
            f"I need to check if the repo contains a the file '{self.filename}'",
        ]


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"
    result: str = "3.9"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(result="3.7"),
            cls(result="3.8"),
        ]

    def use_when(self) -> List[str]:
        return [
            "I need to know which version of Python is needed.",
            "I want to check the Python version.",
            "Is there a specific version of Python",
            "What version of Python should be used",
            "What version of Python",
        ]


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
        return [
            "Here is a sample Dockerfile",
            "You can modify this Dockerfile",
            "Does this look good to you",
            "Here is the Dockerfile",
            "This Dockerfile installs",
            "the above Dockerfile",
            "I will create a Dockerfile",
            "review the proposed Dockerfile",
        ]


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
            "we need to specify the dependencies",
            "Can you tell me the dependencies used in the repo",
        ]
