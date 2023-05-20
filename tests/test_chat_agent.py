from llmagent.agent.chat_agent import ChatAgent
from llmagent.agent.base import AgentConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig


class TestChatAgentConfig(AgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: LLMConfig = LLMConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    )
    parsing: ParsingConfig = None
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


def test_chat_agent():
    cfg = TestChatAgentConfig()
    # just testing that these don't fail
    agent = ChatAgent(cfg)
    response = agent.respond("what is the capital of France?")
    assert "Paris" in response.content
