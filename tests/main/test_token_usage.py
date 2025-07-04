import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig

MAX_OUTPUT_TOKENS = 30


class _TestChatAgentConfig(ChatAgentConfig):
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

    def handle(self):
        return f"The capital of {self.entity} is {self.capital}"


# Define the configurations
config = OpenAIGPTConfig(
    cache_config=RedisCacheConfig(fake=True),
    use_chat_for_completion=True,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    min_output_tokens=1,
)


@pytest.mark.parametrize("stream", [True, False])
def test_agent_token_usage(stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig(llm=config)
    agent = ChatAgent(cfg)
    agent.llm.reset_usage_cost()
    question = "What is the capital of Canada?"
    q_tokens = agent.num_tokens(question)
    agent.llm_response_forget(question)
    assert agent.total_llm_token_usage != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token_usage

    set_global(Settings(cache=True, stream=stream))
    # this convo shouldn't change the cost and tokens because `cache` is `True`
    response0 = agent.llm_response_forget(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd

    # This convo should change the cost because `cache` is `False`:
    # IF the response is identical to before, then the
    # number of accumulated tokens should be doubled, but
    # we allow for variation in the response
    set_global(Settings(cache=False, stream=stream))
    response1 = agent.llm_response(question)
    assert (
        agent.total_llm_token_usage
        == 2 * total_tokens_after_1st_rnd
        + agent.num_tokens(response1.content)
        - agent.num_tokens(response0.content)
    )
    assert agent.total_llm_token_cost > total_cost_after_1st_rnd * 1.1

    # check that cost/usage accumulation in agent matches that in llm
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
        + q_tokens
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
    agent.enable_message(CapitalTool, use=True, handle=True)

    question = "What is the capital of China?"
    response1 = agent.llm_response(question)
    result = agent.agent_response(response1)
    agent.llm_response(result)
    response3 = agent.llm_response(question)

    assert (
        response3.metadata.usage.prompt_tokens
        >= response1.metadata.usage.prompt_tokens
        + response1.metadata.usage.completion_tokens
        + agent.num_tokens(question)
        + agent.num_tokens(result.content)
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
    response0 = await agent.llm_response_forget_async(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token_usage == total_tokens_after_1st_rnd

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be doubled because the question/response pair
    # is the same
    set_global(Settings(cache=False, stream=stream))
    response1 = await agent.llm_response_async(question)
    print("***3rd round***")

    assert (
        agent.total_llm_token_usage
        == 2 * total_tokens_after_1st_rnd
        + agent.num_tokens(response1.content)
        - agent.num_tokens(response0.content)
    )
    assert agent.total_llm_token_cost > total_cost_after_1st_rnd * 1.1

    llm_usage = agent.llm.usage_cost_dict[agent.config.llm.chat_model]
    assert (
        llm_usage.prompt_tokens + llm_usage.completion_tokens
        == agent.total_llm_token_usage
    )
    assert llm_usage.cost == agent.total_llm_token_cost


def test_cached_tokens_tracking():
    """Test that cached tokens are properly tracked in token usage"""
    set_global(Settings(cache=False, stream=False))
    cfg = _TestChatAgentConfig(llm=config)
    agent = ChatAgent(cfg)
    agent.llm.reset_usage_cost()

    # First request - no cached tokens expected
    question = "What is 2+2?"
    response1 = agent.llm_response(question)
    usage1 = response1.metadata.usage
    assert usage1.cached_tokens == 0
    assert usage1.prompt_tokens > 0
    assert usage1.completion_tokens > 0

    # Check cost calculation with no cached tokens
    cost1 = agent.compute_token_cost(usage1.prompt_tokens, 0, usage1.completion_tokens)
    assert cost1 > 0

    # Check cost calculation with simulated cached tokens
    # If half the prompt tokens were cached, cost should be lower
    simulated_cached = usage1.prompt_tokens // 2
    cost_with_cache = agent.compute_token_cost(
        usage1.prompt_tokens, simulated_cached, usage1.completion_tokens
    )
    # With cached tokens, cost should be less (assuming cached cost < input cost)
    assert cost_with_cache < cost1


def test_cached_tokens_in_llm_response():
    """Test that LLMTokenUsage properly includes cached_tokens field"""
    from langroid.language_models.base import LLMTokenUsage

    # Create token usage with cached tokens
    usage = LLMTokenUsage(
        prompt_tokens=100, cached_tokens=25, completion_tokens=50, cost=0.001
    )

    # Verify cached_tokens is tracked
    assert usage.cached_tokens == 25
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50

    # Test string representation includes cached tokens
    usage_str = str(usage)
    assert "cached 25" in usage_str

    # Test reset clears cached tokens
    usage.reset()
    assert usage.cached_tokens == 0
    assert usage.prompt_tokens == 0
