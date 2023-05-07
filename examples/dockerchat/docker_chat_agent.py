from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentMessage


# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"  # name should exactly match method name in agent
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
        Returns:
            str: description of when the message should be used.
        """

        return f"""If you want to know whether a certain file exists in the repo,
                 you have to ask in JSON format to get an accorate answer. For example, 
                 to ask whether file '{self.filename}' exists
                 """


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"

    def use_when(self):
        return "When you want to find out which version of Python is needed"


class DockerChatAgent(ChatAgent):
    def python_version(self, PythonVersionMessage):
        # dummy result for testing: fill in with actual code that calls PyGitHub fn
        # to search for python version in requirements.txt or pyproject.toml, etc.
        return 3.9

    def file_exists(self, message: FileExistsMessage):
        # dummy result, fill with actual code.
        return True if message.filename == "requirements.txt" else False

    # ... other such methods. For each message,
