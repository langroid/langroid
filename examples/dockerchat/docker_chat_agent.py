from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage


# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"  # name should exactly match method name in agent
    # below will be fields that will be used by the agent method to handle the message.
    filename: str = "test.txt"

    def use_when(self):
        """
        Return a string describing when the message should be used, possibly
        parameterized by the field values. This should be a valid english phrase for
        example,
        - "To check whether the number 3 is smaller than 4", or
        - "When you want to check whether file foo.txt exists"
        The returned phrase P should be such that the extended phrase
        "{P}, write the JSON string: ..." is a valid instruction for the LLM.
        Do NOT include the JSON string for the message in this string, that will be
        done automatically by the methods in AgentMessage class.
        Returns:
            str: description of when the message should be used.
        """

        return f"""If you want to know whether a certain file exists in the repo,
                 you have to ask in JSON format to get an accorate answer. For example, 
                 to ask whether file '{self.filename}' exists
                 """
        # notice how this sentence can be extended by
        # ", you should write in this JSON format: "


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"

    def use_when(self):
        return "When you want to find out which version of Python is needed"


class DockerfileMessage(AgentMessage):
    request: str = "dockerfile"
    contents: str = """
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
        """  # contents of dockerfile

    def use_when(self):
        return """
        When you want to show me a dockerfile you have created, send it in 
        JSON Format. For example, to send me a dockerfile with the contents
        {self.contents}
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
