import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


@pytest.mark.parametrize("stream", [True, False])
def test_task(stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    task = Task(agent, name="Test", llm_delegate=False, single_round=False)
    question = "What is the capital of Canada?"
    task.run(msg=question, turns=1)

    assert agent.total_llm_token != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token

    agent.message_history.clear()
    set_global(Settings(cache=True, stream=stream))
    print("***2nd round***")
    # this convo shouldn't change the cost because `cache` is `True`
    # while number of tokens should change
    task.run(msg=question, turns=1)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token != total_tokens_after_1st_rnd

    total_tokens_after_2nd_rnd = agent.total_llm_token

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be based on `prev_tokens` and messages in
    # the message_history
    agent.message_history.clear()
    set_global(Settings(cache=False, stream=stream))
    print("***3rd round***")
    task.run(msg=question, turns=1)
    assert (
        agent.total_llm_token == total_tokens_after_2nd_rnd + total_tokens_after_1st_rnd
    )


@pytest.mark.parametrize("stream", [True, False])
def test_agent(stream):
    set_global(Settings(cache=False, stream=stream))
    cfg = _TestChatAgentConfig()
    agent = ChatAgent(cfg)
    question = "What is the capital of Canada?"
    agent.llm_response_forget(question)
    assert agent.total_llm_token != 0
    assert agent.total_llm_token_cost != 0

    total_cost_after_1st_rnd = agent.total_llm_token_cost
    total_tokens_after_1st_rnd = agent.total_llm_token

    set_global(Settings(cache=True, stream=stream))
    print("***2nd round***")
    # this convo shouldn't change the cost because `cache` is `True`
    # while number of tokens should change
    agent.llm_response_forget(question)
    assert total_cost_after_1st_rnd == agent.total_llm_token_cost
    assert agent.total_llm_token > total_tokens_after_1st_rnd

    total_tokens_after_2nd_rnd = agent.total_llm_token

    # this convo should change the cost because `cache` is `False`
    # number of accumulated tokens should be based on `prev_tokens` and messages in
    # the message_history
    set_global(Settings(cache=False, stream=stream))
    agent.llm_response(question)
    print("***3rd round***")
    assert (
        agent.total_llm_token == total_tokens_after_2nd_rnd + total_tokens_after_1st_rnd
    )
