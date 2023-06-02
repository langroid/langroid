from llmagent.agent.chat_agent import ChatAgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.utils.configuration import Settings, set_global
from llmagent.agent.chat_agent import ChatAgent
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.agent.base import AgentMessage
from llmagent.parsing.parser import ParsingConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig

from typing import List


CONTAINER_RUN_RESPONSE = "Container runs successfully"


class RunContainerMessage(AgentMessage):
    request: str = "run_container"
    purpose: str = """Verify that the container works correctly and preserves 
    the intended behavior.  
    """
    cmd: str = "python"
    tests: List[str] = ["tests/t1.py"]
    result: str = "The container runs correctly"

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Return a list of example messages of this type, for use in testing.
        Returns:
            List[AgentMessage]: list of example messages of this type
        """
        return [
            cls(
                cmd="python",
                tests=["tests/t1.py"],
                result="Container works successfully.",
            ),
            cls(
                cmd="python",
                tests=["tests/t1.py", "tests/t2.py"],
                result="Test case t2 has failed.",
            ),
        ]


class MessageHandlingAgent(ChatAgent):
    def run_container(self, RunContainerMessage) -> str:
        return CONTAINER_RUN_RESPONSE


RUN_CONTAINER_MSG = """
Ok, thank you.
{
'request': 'run_container',
'cmd': 'python',
'tests': 't1.py'
}
uses these arguments to test the container
"""

cfg = ChatAgentConfig(
    name="test-llmagent",
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


def test_enable_message():
    agent.enable_message(RunContainerMessage)
    assert "run_container" in agent.handled_classes
    assert agent.handled_classes["run_container"] == RunContainerMessage


def test_disable_message():
    agent.enable_message(RunContainerMessage)

    agent.disable_message(RunContainerMessage)
    assert "run_container" not in agent.handled_classes


def test_dockerchat_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(RunContainerMessage)
    agent.disable_message(RunContainerMessage)
    assert agent.handle_message(RUN_CONTAINER_MSG) is None


def test_llm_agent_message(test_settings: Settings):
    """
    Test whether LLM is able to generate message in required format, and the
    agent handles the message correctly.
    """
    set_global(test_settings)
    agent = MessageHandlingAgent(cfg)
    agent.enable_message(RunContainerMessage)

    llm_msg = agent.llm_response_forget(
        """Start by asking me about verifying the proposed_dockerfile by 
        running a container based on this Dockerfile."""
    ).content

    agent_result = agent.handle_message(llm_msg)
    assert agent_result == CONTAINER_RUN_RESPONSE
