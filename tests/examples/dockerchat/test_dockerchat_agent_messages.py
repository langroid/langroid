from langroid.agent.chat_agent import ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    FileExistsMessage,
)

from langroid.cachedb.redis_cachedb import RedisCacheConfig
from typing import Optional


cfg = ChatAgentConfig(
    debug=False,
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
)

NONE_MSG = "nothing to see here"

URL_MSG = "A messaging containing the word URL"

ASK_URL_RESPONSE = """You have not yet sent me the URL. 
            Start by asking for the URL, then confirm the URL with me"""

GOT_URL_RESPONSE = """
Ok, confirming the URL. 
"""

FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists",
"filename": "___test.txt"
} 
Hope you can tell me!
"""

ASK_URL_MSG = """
great, let me know the URL:
{
"request": "ask_url"
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


def test_dockerchat_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(FileExistsMessage)
    agent.enable_message(AskURLMessage)
    # any msg before ask_url will result in an agent response
    # telling LLM to ask for URL
    assert agent.handle_message(NONE_MSG) == ASK_URL_RESPONSE
    assert agent.handle_message(FILE_EXISTS_MSG) == ASK_URL_RESPONSE
    assert agent.handle_message(ASK_URL_MSG) == GOT_URL_RESPONSE

    agent.disable_message_handling(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
