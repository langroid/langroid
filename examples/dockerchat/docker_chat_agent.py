from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage
from examples.dockerchat.identify_python_version import get_python_version
from examples.dockerchat.identify_python_dependency import (
    identify_dependency_management,
)
from typing import List


import subprocess
import os
import logging

logger = logging.getLogger(__name__)

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


class DockerChatAgent(ChatAgent):
    repo_path: str = "/nobackup/images_repos/Auto-GPT"

    def python_version(self, PythonVersionMessage) -> str:
        """
        Identifies Python version for a given repo
        Args:
        Returns:
            str: a string indicates the identified python version or indicate the version can't be identified
        """
        python_version = get_python_version(self.repo_path)
        if python_version:
            return python_version
        else:
            logger.error("Could not determine Python version.")
        return "Couldn't identify the python version"

    def file_exists(self, message: FileExistsMessage) -> str:
        # dummy result, fill with actual code.
        if message.filename == "requirements.txt":
            return f"""
            Yes, there is a file named {message.filename} in the repo."""
        else:
            return f"""
            No, there is no file named {message.filename} in the repo."""

    def python_dependency(self, PythonDependencyMessage) -> str:
        """
        Identifies Python dependencies in a given repo by inspecting various artifacts like requirements.txt
        Args:
        Returns:
            str: a string indicates the identified the dependency management approach
        """
        python_dependency = identify_dependency_management(self.repo_path)
        if python_dependency:
            return f"Dependencies in this repo are managed using: {python_dependency}"
        else:
            return "Dependencies are not defined in this repo"

    def _cleanup_dockerfile(self, img_name: str, dockerfile_path: str) -> None:
        """
        Remove Dockefile and built image after performing the verification process
        Args:
            img_name (str): the name of the Docker image
            dockerfile_path (str): path to the saved Dockerfile
        """
        try:
            # Remove Dockerfile
            if os.path.exists(dockerfile_path):
                os.remove(dockerfile_path)
                logger.info(f"Dockerfile at path '{dockerfile_path}' has been removed.")
            else:
                logger.error(f"No Dockerfile found at path '{dockerfile_path}'.")

            # Remove Docker image
            command = f"docker rmi -f {img_name}"
            result = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Check if the command was successful
            if result.returncode == 0:
                logger.info(f"Docker image '{img_name}' has been removed.")
            else:
                logger.error(
                    f"Failed to remove Docker image '{img_name}'. Error: {result.stderr.decode()}"
                )

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def _save_dockerfile(self, dockerfile: str) -> str:
        """
        Save the proposed Dockerfile in the root directory of a repo
        Args:
            dockerfile (str): content of the dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been saved successfully
        """
        try:
            full_path = os.path.join(self.repo_path, "Dockerfile")
            with open(full_path, "w") as f:
                f.write(dockerfile)
            return full_path
        except Exception as e:
            return f"An error occurred while saving the Dockerfile: {e}"

    def validate_dockerfile(self, dockerfile_msg: ValidateDockerfileMessage) -> str:
        """
        validates the proposed Dockerfile by LLM. The validation process involves saving the proposed_dockerfile, building the image, and finally cleanning up
        Args:
            dockerfile_msg (ValidateDockerfileMessage): LLM message contains the definition of the Dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been built successfully
        """
        proposed_dockerfile = dockerfile_msg.proposed_dockerfile
        try:
            dockerfile_path = self._save_dockerfile(proposed_dockerfile)
            if dockerfile_path.startswith("An error"):
                return dockerfile_path

            original_path = os.getcwd()
            os.chdir(self.repo_path)

            # Build the Docker image
            img_name = "validate_img"
            command = f"docker build -t {img_name} -f {dockerfile_path} ."
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            os.chdir(original_path)

            # Check the result of the build process
            if process.returncode == 0:
                # do some cleaning: remove the Docker image and the Dockerfile
                self._cleanup_dockerfile(img_name, dockerfile_path)
                return "Docker build was successful"
            else:
                return f"Docker build failed with error message: {process.stderr}"

        except Exception as e:
            return f"An error occurred during the Docker build: {e}"
