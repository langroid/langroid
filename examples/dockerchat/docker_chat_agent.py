from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage


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
