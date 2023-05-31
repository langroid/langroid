from llmagent.agent.base import AgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.dockerchat_agent_messages import RunContainerMessage
from llmagent.utils.configuration import update_global_settings, Settings, set_global
from llmagent.agent.chat_agent import ChatAgent
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig


CONTAINER_RUN_RESPONSE = "Container runs successfully"


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

cfg = AgentConfig(
    debug=True,
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
    update_global_settings(cfg, keys=["debug"])
    agent = MessageHandlingAgent(cfg)
    agent.enable_message(RunContainerMessage)

    llm_msg = agent.respond_forget(
        """Start by asking me about verifying the proposed_dockerfile by running a container based on this Dockerfile."""
    ).content

    agent_result = agent.handle_message(llm_msg)
    assert agent_result == CONTAINER_RUN_RESPONSE
