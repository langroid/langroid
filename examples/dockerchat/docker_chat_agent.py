from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage
from typing import List

import subprocess
import os

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

    @staticmethod
    def cleanup_dockerfile(img_name: str, dockerfile_path: str) -> None:
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
                print(f"Dockerfile at path '{dockerfile_path}' has been removed.")
            else:
                print(f"No Dockerfile found at path '{dockerfile_path}'.")

            # Remove Docker image
            command = f"docker rmi -f {img_name}"
            result = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Check if the command was successful
            if result.returncode == 0:
                print(f"Docker image '{img_name}' has been removed.")
            else:
                print(
                    f"Failed to remove Docker image '{img_name}'. Error: {result.stderr.decode()}"
                )

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    @staticmethod
    def save_dockerfile(dockerfile: str, repo_path: str) -> str:
        """
        Save the proposed Dockerfile in the root directory of a repo
        Args:
            dockerfile (str): content of the dockerfile
            repo_path (str): path to the cloned repo
        Returns:
            str: a string indicates whether the Dockerfile has been saved successfully
        """
        try:
            full_path = os.path.join(repo_path, "Dockerfile")
            with open(full_path, "w") as f:
                f.write(dockerfile)
            return full_path
        except Exception as e:
            return f"An error occurred while saving the Dockerfile: {e}"

    @staticmethod
    def build_docker_image(message: str, repo_path: str) -> str:
        """
        validates the proposed Dockerfile by LLM.
        Args:
            message (DockerfileMessage): LLM message contains the Dockerfile
            repo_path (str): path to the cloned repo
        Returns:
            str: a string indicates whether the Dockerfile has been built successfully
        """
        try:
            dockerfile_path = DockerChatAgent.save_dockerfile(message, repo_path)
            if dockerfile_path.startswith("An error"):
                return dockerfile_path

            # Build the Docker image
            img_name = "verify_img"
            command = f"docker build -t {img_name} -f {dockerfile_path} ."
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check the result of the build process
            if process.returncode == 0:
                # do some cleaning: the Docker image and remove the Dockerfile
                DockerChatAgent.cleanup_dockerfile(img_name, dockerfile_path)
                return "Docker build was successful"
            else:
                return f"Docker build failed with error message: {process.stderr}"

        except Exception as e:
            return f"An error occurred during the Docker build: {e}"
