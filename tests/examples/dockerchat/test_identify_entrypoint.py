import tempfile
import os
from typing import List
import json
import pytest

from llmagent.utils.system import rmdir
from examples.dockerchat.identify_docker_entrypoint_cmd import identify_entrypoint_CMD
from llmagent.agent.base import AgentMessage
from llmagent.agent.base import AgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.agent.chat_agent import ChatAgent
from llmagent.utils.configuration import update_global_settings
from llmagent.language_models.base import Role, LLMMessage


class EntryPointAndCMDMessage(AgentMessage):
    request: str = "get_entrypoint_cmd"
    purpose: str = "To define commands for ENTRYPOINT, CMD, both, or none."
    result: str = "yes"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(result="This repo uses this command for both ENTRYPOINT and CMD"),
            cls(
                result="This repo uses this command for ENTRYPOINT and this command for CMD"
            ),
            cls(result="This repo doesn't use ENTRYPOINT"),
            cls(result="This repo doesn't use CMD"),
        ]


class MessageHandlingAgent(ChatAgent):
    def get_entrypoint_cmd(
        self, EntryPointAndCMDMessage, cmd: bool = False, entrypoint: bool = False
    ) -> str:
        return {"entrypoint": '["python", "main.py"]', "cmd": None}


qd_dir = ".qdrant/testdata_test_agent"
rmdir(qd_dir)
cfg = AgentConfig(
    debug=True,
    name="test-llmagent",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=None,
    prompts=PromptsConfig(),
)
agent = MessageHandlingAgent(cfg)


def test_enable_message():
    agent.enable_message(EntryPointAndCMDMessage)
    assert "get_entrypoint_cmd" in agent.handled_classes
    assert agent.handled_classes["get_entrypoint_cmd"] == EntryPointAndCMDMessage


def test_disable_message():
    agent.enable_message(EntryPointAndCMDMessage)
    agent.disable_message(EntryPointAndCMDMessage)
    assert "python_dependency" not in agent.handled_classes


@pytest.mark.parametrize("msg_cls", [EntryPointAndCMDMessage])
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls().usage_example()
    assert json.loads(usage)["request"] == msg_cls().request


rmdir(qd_dir)  # don't need it here


NONE_MSG = "nothing to see here"

ENTRYPOINT_CMD_MSG = """
great, please tell me this --
{
'request': 'get_entrypoint_cmd'
}/if you know it
"""


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(EntryPointAndCMDMessage)
    assert agent.handle_message(NONE_MSG) is None
    a, b = agent.handle_message(ENTRYPOINT_CMD_MSG)
    assert agent.handle_message(ENTRYPOINT_CMD_MSG) == {
        "entrypoint": '["python", "main.py"]',
        "cmd": None,
    }

    agent.disable_message(EntryPointAndCMDMessage)
    assert agent.handle_message(ENTRYPOINT_CMD_MSG) is None

    agent.enable_message(EntryPointAndCMDMessage)
    assert agent.handle_message(ENTRYPOINT_CMD_MSG) == {
        "entrypoint": '["python", "main.py"]',
        "cmd": None,
    }


def test_llm_agent_message():
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    update_global_settings(cfg, keys=["debug"])
    task_messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""
            You are a devops engineer, and your task is to understand a PYTHON 
            repo. Plan this out step by step, and ask me questions 
            for any info you need to understand the repo. 
            """,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
            You are an assistant whose task is to understand a Python repo.

            You have to think in small steps, and at each stage, show me your 
            THINKING, and the QUESTION you want to ask. Based on my answer, you will 
            generate a new THINKING and QUESTION.  
            """,
        ),
    ]
    agent = MessageHandlingAgent(cfg, task_messages)
    agent.enable_message(EntryPointAndCMDMessage)

    agent.run(
        iters=2, default_human_response="I don't know, please ask your next question."
    )


def test_identify_entrypoint_CMD():
    directory = ""
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        directory = tmp_dir
        main_file = open("main.py", "w")
        main_file.write(
            """
        def main():
            print("This is the main function.")

        if __name__ == "__main__":
            main()
        """
        )
        main_file.close()

        other_file = open("other.py", "w")
        other_file.write(
            """
        def other():
            print("This is another function.")
        """
        )
        other_file.close()

        cmd, entrypoint = identify_entrypoint_CMD(directory, cmd=True, entrypoint=True)
        assert cmd == ["arg1", "arg2"]
        assert entrypoint == ["python", "main.py"]

        cmd, entrypoint = identify_entrypoint_CMD(directory, cmd=True, entrypoint=False)
        assert cmd == ["python", "main.py"]
        assert entrypoint == ["/bin/sh", "-c"]

        cmd, entrypoint = identify_entrypoint_CMD(directory, cmd=False, entrypoint=True)
        assert cmd is None
        assert entrypoint == ["python", "main.py"]

        cmd, entrypoint = identify_entrypoint_CMD(
            directory, cmd=False, entrypoint=False
        )
        assert cmd is None
        assert entrypoint == ["/bin/sh", "-c"]
