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
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        use_chat_for_completion=True,
        cache_config=RedisCacheConfig(fake=False),
    )
    parsing: ParsingConfig = ParsingConfig()

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


def test_usage_dict_cache_true():
    """
    If cache is True, then total tokens should be zero.
    """
    set_global(Settings(cache=True))
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response_no_stream = agent.llm_response("what is the capital of France?")
    assert response_no_stream.metadata.usage["total_tokens"] == 0
    with StreamingIfAllowed(agent.llm, False):
        response_stream = agent.llm_response("what is the capital of France?")
    assert response_stream.metadata.usage["total_tokens"] == 0
    assert (
        response_stream.metadata.usage["total_tokens"]
        == response_no_stream.metadata.usage["total_tokens"]
    )


def test_usage_dict_cache_false():
    """
    If cache is False, then total tokens should not be zero.
    """
    set_global(Settings(cache=False))
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response_no_stream = agent.llm_response("what is the capital of France?")
    assert response_no_stream.metadata.usage["total_tokens"] != 0
    with StreamingIfAllowed(agent.llm, False):
        response_stream = agent.llm_response("what is the capital of France?")
    assert response_stream.metadata.usage["total_tokens"] != 0
    assert (
        response_stream.metadata.usage["total_tokens"]
        == response_no_stream.metadata.usage["total_tokens"]
    )
