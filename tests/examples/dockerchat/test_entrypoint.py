from llmagent.agent.base import AgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    EntryPointAndCMDMessage,
    AskURLMessage
)

from llmagent.agent.message import AgentMessage
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from typing import Optional
import json
import pytest

cfg = AgentConfig(
    debug=False,
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        cache_config=RedisCacheConfig(fake=False),
    ),
)

ASK_URL_RESPONSE = """You have not yet sent me the URL. 
            Start by asking for the URL, then confirm the URL with me"""

GOT_URL_RESPONSE = """
Ok, confirming the URL. 
"""

FIND_ENTRYPOINT_MSG = """
great, please tell me this --
{
'request': 'find_entrypoint'
}/if you know it
"""

ASK_URL_MSG = """
great, let me know the URL:
{
'request': 'ask_url'
}
"""


class _TestDockerChatAgent(DockerChatAgent):
    def handle_message_fallback(self, input_str: str = "") -> Optional[str]:
        # if URL not yet known, tell LLM to ask for it, unless this msg
        # contains the word URL
        if self.repo_path is None and "url" not in input_str.lower():
            return ASK_URL_RESPONSE

    def ask_url(self, msg: AskURLMessage) -> str:
        self.repo_path = "dummy"
        return GOT_URL_RESPONSE


agent = _TestDockerChatAgent(cfg)


def test_enable_message():
    agent.enable_message(EntryPointAndCMDMessage)
    assert "find_entrypoint" in agent.handled_classes
    assert agent.handled_classes["find_entrypoint"] == EntryPointAndCMDMessage


def test_disable_message():
    agent.enable_message(EntryPointAndCMDMessage)

    agent.disable_message(EntryPointAndCMDMessage)
    assert "find_entrypoint" not in agent.handled_classes


@pytest.mark.parametrize("msg_cls", [AskURLMessage, EntryPointAndCMDMessage])
def test_usage_instruction(msg_cls: AgentMessage):
    usage = msg_cls().usage_example()
    assert json.loads(usage)["request"] == msg_cls().request


def test_dockerchat_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(EntryPointAndCMDMessage)
    agent.enable_message(AskURLMessage)
    # any msg before ask_url will result in an agent response
    # telling LLM to ask for URL
    assert agent.handle_message(ASK_URL_MSG) == GOT_URL_RESPONSE
    
    agent.disable_message(EntryPointAndCMDMessage)
    assert agent.handle_message(FIND_ENTRYPOINT_MSG) is None
