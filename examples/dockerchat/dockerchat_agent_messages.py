from llmagent.agent.base import AgentMessage
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class AskURLMessage(AgentMessage):
    request: str = "ask_url"
    purpose: str = "To get the github repo url from the user."
    result: str = ""

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(result="https://github.com/hello/world"),
        ]


class RunPythonMessage(AgentMessage):
    request: str = "run_python"
    purpose: str = """
    To run a python <code> on the repository, to find desired info.
    This code can assume it has access to the code repository in a local folder.
    The code can be as detailed as needed, including import statements.
    This tool can be used to find any info not available from the other tools.
    """
    code: str
    result: str = ""

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(code="print('hello world')", result="hello world"),
        ]


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"  # name should exactly match method name in agent
    # below will be fields that will be used by the agent method to handle the message.
    purpose: str = "To check if a file <filename> exists in the repo."
    filename: str
    result: str = "no"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(filename="blah.md", result="yes"),
        ]


class ValidateDockerfileMessage(AgentMessage):
    request: str = "validate_dockerfile"
    purpose: str = """
    To show a <proposed_dockerfile> to the user. Use this tool whenever you want 
    to SHOW or VALIDATE a <proposed_dockerfile>. NEVER list out a dockerfile 
    without using this tool
    """

    proposed_dockerfile: Union[str, List[str]]
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


class PythonDependencyMessage(AgentMessage):
    request: str = "python_dependency"
    purpose: str = "To find out where python dependencies are listed."
    result: str = "This repo uses requirements.txt for managing dependencies"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(
                result="This repo uses requirements.txt for managing dependencies",
            ),
            cls(
                result="This repo uses pyproject.toml for managing dependencies",
            ),
            cls(result="This repo doesn't contain any dependacy manager"),
        ]


class EntryPointAndCMDMessage(AgentMessage):
    request: str = "find_entrypoint"
    purpose: str = """To identify main scripts and their arguments that can 
    be used for ENTRYPOINT, CMD, both, or none."""
    result: str = "The main script is main.py"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(
                result="""The name of the main script in this repo is main.py. 
                To run it, you can use the command python main.py
                """
            ),
            cls(result="I could not find the entry point."),
            cls(result="This repo doesn't have main script"),
        ]


class RunContainerMessage(AgentMessage):
    request: str = "run_container"
    purpose: str = """Verify that the container works correctly and preserves 
    the intended behavior. This will use the image built using 
    the proposed dockerfile and EXPECTS to receive <test>.
    <test> is a command and the test case. <location> indicates whether 
    the <test> should be executed from INSIDE or OUTSIDE the container. <run> 
    is a docker run command with all required arguments that will be used to 
    run the container, where image_name is `validate_img:latest`
    """
    test: str
    location: str
    run: str
    result: str = "Inside test case works successfully."

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(
                test="python tests/t1.py",
                location="Inside",
                run="docker run",
                result="Inside test case works successfully.",
            ),
            cls(
                test="curl localhost",
                location="Outside",
                run="docker run",
                result="Outside test case has failed.",
            ),
        ]
