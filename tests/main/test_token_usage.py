import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


class CapitalTool(ToolMessage):
    request = "capital"
    purpose = "To present the <capital> of an <entity> (state or country)"
    entity: str
    capital: str


# Define the configurations
config = OpenAIGPTConfig(
    cache_config=RedisCacheConfig(fake=False),
    chat_model=OpenAIChatModel.GPT4_TURBO,
    use_chat_for_completion=True,
)


@pytest.mark.parametrize("stream", [True, False])
def test_agent_token_usage(stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig(llm=config)
    agent = ChatAgent(cfg)
    agent.llm.reset_usage_cost()
    question = "What is the capital of Canada?"
    agent.llm_response_forget(question)
    assert agent.total_llm_token_usage != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token_usage

    set_global(Settings(cache=True, stream=stream))
    # this convo shouldn't change the cost and tokens because `cache` is `True`
    agent.llm_response_forget(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be doubled because the question/response pair
    # is the same
    set_global(Settings(cache=False, stream=stream))
    response1 = agent.llm_response(question)
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd * 2
    assert agent.total_llm_token_cost == total_cost_after_1st_rnd * 2
    llm_usage = agent.llm.usage_cost_dict[agent.config.llm.chat_model]
    assert (
        llm_usage.prompt_tokens + llm_usage.completion_tokens
        == agent.total_llm_token_usage
    )
    assert llm_usage.cost == agent.total_llm_token_cost

    # check proper accumulation of prompt tokens across multiple rounds
    response2 = agent.llm_response(question)
    assert (
        response2.metadata.usage.prompt_tokens
        >= response1.metadata.usage.prompt_tokens
        + response1.metadata.usage.completion_tokens
        + agent.num_tokens(question)
    )


@pytest.mark.parametrize("fn", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_token_usage_tool(fn, stream):
    """Check token usage accumulation with tool/function-call"""
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig(
        llm=config,
        use_functions_api=fn,
        use_tools=not fn,
        system_message="Use the `capital` tool to tell me the capital of a country",
    )
    agent = ChatAgent(cfg)
    agent.llm.reset_usage_cost()
    agent.enable_message(CapitalTool, use=True, handle=False)

    question = "What is the capital of China?"
    response1 = agent.llm_response(question)
    response2 = agent.llm_response(question)

    assert (
        response2.metadata.usage.prompt_tokens
        >= response1.metadata.usage.prompt_tokens
        + response1.metadata.usage.completion_tokens
        + agent.num_tokens(question)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_agent_token_usage_async(stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig(llm=config)
    agent = ChatAgent(cfg)
    agent.llm.reset_usage_cost()
    question = "What is the capital of Canada?"
    await agent.llm_response_forget_async(question)
    assert agent.total_llm_token_usage != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token_usage

    set_global(Settings(cache=True, stream=stream))
    print("***2nd round***")
    # this convo shouldn't change the cost and tokens because `cache` is `True`
    await agent.llm_response_forget_async(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be doubled because the question/response pair
    # is the same
    set_global(Settings(cache=False, stream=stream))
    await agent.llm_response_async(question)
    print("***3rd round***")

    b = max(agent.total_llm_token_usage, total_tokens_after_1st_rnd * 2)
    assert abs(agent.total_llm_token_usage - total_tokens_after_1st_rnd * 2) < 0.1 * b

    b = max(agent.total_llm_token_cost, total_cost_after_1st_rnd * 2)
    assert abs(agent.total_llm_token_cost - total_cost_after_1st_rnd * 2) < 0.1 * b

    llm_usage = agent.llm.usage_cost_dict[agent.config.llm.chat_model]
    assert (
        llm_usage.prompt_tokens + llm_usage.completion_tokens
        == agent.total_llm_token_usage
    )
    assert llm_usage.cost == agent.total_llm_token_cost
