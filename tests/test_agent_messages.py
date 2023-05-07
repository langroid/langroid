from llmagent.agent.base import AgentConfig
from llmagent.language_models.base import Role, LLMMessage
from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.message import AgentMessage
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.system import rmdir
import pytest


class FileExistsMessage(AgentMessage):
    request: str = "file_exists"
    filename: str = "test.txt"

    def use_when(self):
        return f"""If you want to know whether a certain file exists in the repo,
                 you have to ask in JSON format to get an accorate answer. For example, 
                 to ask whether file '{self.filename}' exists
                 """


class PythonVersionMessage(AgentMessage):
    request: str = "python_version"

    def use_when(self):
        return "When you want to find out which version of Python is needed"


class MessageHandlingAgent(ChatAgent):
    def file_exists(self, message: FileExistsMessage):
        return True if message.filename == "requirements.txt" else False

    def python_version(self, PythonVersionMessage):
        return 3.9


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = AgentConfig(
    name="test-llmagent",
    debug=False,
    vecdb=QdrantDBConfig(
        type="qdrant",
        collection_name="test",
        storage_path=qd_dir,
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    ),
    llm=LLMConfig(
        type="openai",
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


def test_enable_message():
    agent.enable_message(FileExistsMessage)
    assert "file_exists" in agent.handled_classes
    assert agent.handled_classes["file_exists"] == FileExistsMessage

    agent.enable_message(PythonVersionMessage)
    assert "python_version" in agent.handled_classes
    assert agent.handled_classes["python_version"] == PythonVersionMessage


def test_disable_message():
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)

    agent.disable_message(FileExistsMessage)
    assert "file_exists" not in agent.handled_classes

    agent.disable_message(PythonVersionMessage)
    assert "python_version" not in agent.handled_classes


def test_usage_instruction():
    usage = PythonVersionMessage().usage_instruction()
    assert PythonVersionMessage().use_when() in usage

    usage = FileExistsMessage().usage_instruction()
    assert FileExistsMessage().use_when() in usage


rmdir(qd_dir)  # don't need it here


@pytest.mark.parametrize(
    "message, expected",
    [
        ("nothing to see here", None),
        (
            """Ok, thank you. 
                {
                'request': 'file_exists',
                'filename': 'test.txt'
                } 
                Hope you can tell me!
                """,
            False,
        ),
        (
            """great, please tell me this --
                {
                'request': 'python_version'
                }/if you know it
                """,
            3.9,
        ),
    ],
)
def test_agent_actions(message, expected):
    """
    Test whether messages are handled correctly.
    """
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    result = agent.handle_message(message)
    assert result == expected


def test_llm_agent_message():
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    instructions = agent.message_instructions()
    task = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""you are a devops engineer, trying to understand a Python repo""",
        ),
        LLMMessage(
            role=Role.USER,
            content="""You are a devops engineer, trying to understand a Python repo  
        can ask me questions about the repo, one at a time, and I will try to answer.
        """,
        ),
        LLMMessage(
            role=Role.USER,
            content=instructions,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
        Now start asking me questions!
        Remember you have to ask in JSON format if it fits one of the cases above.
        """,
        ),
    ]
    agent.task_messages = task
    ntries = 0
    max_tries = 4
    # we can't predict what the LLM would start with, so we just keep re-trying until
    # it makes the "file_exists" request
    llm_response = agent.start().content
    while True:
        ntries += 1
        if ntries > max_tries:
            break
        result = agent.handle_message(llm_response)
        if result is None:
            llm_response = agent.respond(
                """
            Please check if your message fits one of the 
            JSON examples above, and if so, resend as JSON. Otherwise 
            repeat your previous response
            """
            ).content
            result = agent.handle_message(llm_response)
        if result is None:
            assert (
                "file_exists" not in llm_response
                and "python_version" not in llm_response
            )
            result = "I don't know, please ask your next question."
        else:
            assert "file_exists" in llm_response or "python_version" in llm_response
            result = f"{result}"
        llm_response = agent.respond(result).content
