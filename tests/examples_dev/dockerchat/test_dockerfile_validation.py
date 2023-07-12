import json
from typing import List

from langroid.agent.base import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.system import rmdir


class ValidateDockerfileMessage(ToolMessage):
    request: str = "validate_dockerfile"
    purpose: str = "To check whether a Dockerfile is valid."
    proposed_dockerfile: str = """
        # Use an existing base image
        FROM ubuntu:latest
        # Set the maintainer information
        LABEL maintainer="your_email@example.com"
        # Set the working directory
        """  # contents of dockerfile
    result: str = "build succeed"

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
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


class MessageHandlingAgent(ChatAgent):
    def validate_dockerfile(self, ValidateDockerfileMessage) -> str:
        return "Built successfully"


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = ChatAgentConfig(
    system_message="""
    You are a devops engineer, and your task is to understand a PYTHON
    repo. Plan this out step by step, and ask me questions
    for any info you need to understand the repo.
    """,
    user_message="""
    You are an assistant whose task is to understand a Python repo.

    You have to think in small steps, and at each stage, show me your 
    THINKING, and the QUESTION you want to ask. Based on my answer, you will 
    generate a new THINKING and QUESTION.  
    """,
    debug=True,
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


rmdir(qd_dir)  # don't need it here

df = "FROM ubuntu:latest\nLABEL maintainer=blah"

df_json = json.dumps(df)

NONE_MSG = "nothing to see here"

VALIDATE_DOCKERFILE_MSG = f"""
Ok, thank you.
{{
"request": "validate_dockerfile",
"proposed_dockerfile": {df_json}
}} 
this is the dockerfile
"""

FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists",
"filename": "test.txt"
} 
Hope you can tell me!
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(ValidateDockerfileMessage)
    assert agent.handle_message(NONE_MSG) is None
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) == "Built successfully"

    agent.disable_message_handling(ValidateDockerfileMessage)
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) is None

    agent.enable_message(ValidateDockerfileMessage)
    assert agent.handle_message(VALIDATE_DOCKERFILE_MSG) == "Built successfully"


def test_llm_tool_message(test_settings: Settings):
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    set_global(test_settings)
    agent = MessageHandlingAgent(cfg)
    agent.enable_message(ValidateDockerfileMessage)
    task = Task(
        agent,
        default_human_response="I don't know, please ask your next question.",
    )
    task.run(turns=2)
    # TODO - need to put an assertion here


def clean_string(string: str) -> str:
    """
    removes whitespace in possibly mutli-line string
    Args:
        s(str): string to be modified
    Returns:
        string after cleaning up whitespace
    """
    pieces = [s.replace("\\n", "") for s in string.split()]
    return "".join(pieces)
