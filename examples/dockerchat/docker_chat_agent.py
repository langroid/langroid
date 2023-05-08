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

    def not_use_when(self):
        return """
        THINKING: I need to know the capital of France.
        QUESTION: What is the capital of France?
        """


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"
    result: str = "3.9"

    def use_when(self):
        return "What is the python version needed for this repo?"

    def not_use_when(self):
        return """
        THINKING: I need to add 3 + 4.
        QUESTION: What is the sum of 3 and 4?
        """


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
        return """
        Here is the dockerfile I have written:
        {self.contents}
        """

    def not_use_when(self):
        return """
        THINKING: I need to know the number of files in the repo.
        QUESTION: How many files are in the repo?
        """


class DockerChatAgent(ChatAgent):
    def python_version(self, PythonVersionMessage) -> str:
        # dummy result for testing: fill in with actual code that calls PyGitHub fn
        # to search for python version in requirements.txt or pyproject.toml, etc.
        return "3.9"

    def file_exists(self, message: FileExistsMessage) -> str:
        # dummy result, fill with actual code.
        return "yes" if message.filename == "requirements.txt" else "no"

    def dockerfile(self, message: DockerfileMessage) -> str:
        # dummy result, fill with actual code., like testing it, etc.
        # The response should be some feedback to LLM on validity, etc.
        return "Dockerfile received and validated"

    # ... other such methods.

    # There should be a 1-1 correspondence between message types and agent methods.
