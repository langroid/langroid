from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage
from typing import List


# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


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
        ]


class DockerfileMessage(AgentMessage):
    request: str = "dockerfile"
    contents: str = """
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
        """  # contents of dockerfile
    result: str = "received, but there are errors"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(
                contents="""
                FROM ubuntu:latest
                LABEL maintainer=blah
                """,
                result="received, but there are errors",
            ),
            cls(
                contents="""
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
            f"I need to show the dockerfile I have written: {self.contents}",
            f"I want to send a dockerfile: {self.contents}",
        ]


class DockerChatAgent(ChatAgent):
    def python_version(self, PythonVersionMessage) -> str:
        # dummy result for testing: fill in with actual code that calls PyGitHub fn
        # to search for python version in requirements.txt or pyproject.toml, etc.
        return "The python version is 3.9."

    def file_exists(self, message: FileExistsMessage) -> str:
        # dummy result, fill with actual code.
        if message.filename == "requirements.txt":
            return f"""
            Yes, there is a file named {message.filename} in the repo."""
        else:
            return f"""
            No, there is no file named {message.filename} in the repo."""

    def dockerfile(self, message: DockerfileMessage) -> str:
        # dummy result, fill with actual code., like testing it, etc.
        # The response should be some feedback to LLM on validity, etc.
        return "Dockerfile received and validated"

    # ... other such methods.

    # There should be a 1-1 correspondence between message types and agent methods.
