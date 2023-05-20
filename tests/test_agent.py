from llmagent.agent.base import Agent, AgentConfig
from llmagent.language_models.base import StreamingIfAllowed
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig


class CustomAgentConfig(AgentConfig):
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = None
    llm: LLMConfig = LLMConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    )
    parsing: ParsingConfig = None

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


def test_agent():
    """
    Test whether the combined configs work as expected.
    """
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response = agent.respond("what is the capital of France?")  # direct LLM question
    assert "Paris" in response.content

    with StreamingIfAllowed(agent.llm, False):
        response = agent.respond("what is the capital of France?")
    assert "Paris" in response.content
