from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage

import os
import subprocess


# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"  # name should exactly match method name in agent
    # below will be fields that will be used by the agent method to handle the message.
    filename: str = "test.txt"
    result: str = "yes"

    def use_when(self):
        """
        Return a string showing an example of when the message should be used, possibly
        parameterized by the field values. This should be a valid english phrase in
        first person, in the form of a question or a request.
        - "I want to know whether the file blah.txt is in the repo"
        - "What is the python version needed for this repo?"
        Returns:
            str: example of a situation when the message should be used.
        """

        return f"""I want to know whether there is a file 
        named '{self.filename}' in the repo. 
        """


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"
    result: str = "3.9"

    def use_when(self):
        return "I need to know which version of Python is needed."


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

    def use_when(self):
        return f"""
        Here is the dockerfile I have written:
        {self.contents}
        """


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

    @classmethod
    def save_dockerfile(cls, dockerfile: str, repo_path: str) -> str:
        try:
            full_path = os.path.join(repo_path, "Dockerfile")
            with open(full_path, 'w') as f:
                f.write(dockerfile)
            return full_path
        except Exception as e:
            return f"An error occurred while saving the Dockerfile: {e}"
    
    @classmethod
    def build_docker_image(cls, message: str, repo_path: str) -> str:
        """
        validates the proposed Dockerfile by LLM.
        Args:
            message (DockerfileMessage): LLM message contains the Dockerfile
            repo_path (str): path to the cloned repo
        Returns:
            str: a string indicates whether the Dockerfile 
        """
        try:
            dockerfile_path = DockerChatAgent.save_dockerfile(message, repo_path)
            if dockerfile_path.startswith("An error"):
                return dockerfile_path
            # Change to the specified directory
            original_path = os.getcwd()
            os.chdir(repo_path)

            # Build the Docker image
            command = f'docker build -t your_image_name -f {dockerfile_path} .'
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Restore the original working directory
            os.chdir(original_path)

            # Check the result of the build process
            if process.returncode == 0:
                return "Docker build was successful"
            else:
                return f"Docker build failed with error message: {process.stderr}"

        except Exception as e:
            return f"An error occurred during the Docker build: {e}"
