import asyncio

import pytest

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


@pytest.mark.asyncio
async def test_agent_async(test_settings: Settings):
    """
    Test whether the combined configs work as expected,
    with async calls.
    """
    set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content

    with StreamingIfAllowed(agent.llm, False):
        response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content


@pytest.mark.asyncio
async def test_agent_async_concurrent(test_settings: Settings):
    set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    # Async calls should work even if the agent is not async

    N = 3
    questions = ["1+" + str(i) for i in range(N)]
    expected_answers = [str(i + 1) for i in range(N)]
    answers = await asyncio.gather(
        *(agent.llm_response_async(question) for question in questions)
    )
    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.content for a in answers)
