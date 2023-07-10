from langroid.agent.base import Agent, AgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import StreamingIfAllowed
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class CustomAgentConfig(AgentConfig):
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
        cache_config=RedisCacheConfig(fake=False),
    )
    parsing: ParsingConfig = ParsingConfig()

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


def test_agent(test_settings: Settings):
    """
    Test whether the combined configs work as expected.
    """
    set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response = agent.llm_response(
        "what is the capital of France?"
    )  # direct LLM question
    assert "Paris" in response.content

    with StreamingIfAllowed(agent.llm, False):
        response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content
