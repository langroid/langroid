import asyncio

import pytest

import langroid as lr
import langroid.language_models as lm


class CustomAgentConfig(lr.AgentConfig):
    max_tokens: int = 10000
    llm: lm.LLMConfig = lm.OpenAIGPTConfig(
        cache_config=lr.cachedb.redis_cachedb.RedisCacheConfig(fake=False),
    )


def test_agent(test_settings: lr.utils.configuration.Settings):
    """
    Test whether the combined configs work as expected.
    """
    lr.utils.configuration.set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = lr.Agent(agent_config)
    response = agent.llm_response(
        "what is the capital of France?"
    )  # direct LLM question
    assert "Paris" in response.content

    with lr.language_models.base.StreamingIfAllowed(agent.llm, False):
        response = agent.llm_response("what is the capital of France?")
    assert "Paris" in response.content


@pytest.mark.asyncio
async def test_agent_async(test_settings: lr.utils.configuration.Settings):
    """
    Test whether the combined configs work as expected,
    with async calls.
    """
    lr.utils.configuration.set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = lr.Agent(agent_config)
    response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content

    with lr.language_models.base.StreamingIfAllowed(agent.llm, False):
        response = await agent.llm_response_async("what is the capital of France?")
    assert "Paris" in response.content


@pytest.mark.asyncio
async def test_agent_async_concurrent(test_settings: lr.utils.configuration.Settings):
    lr.utils.configuration.set_global(test_settings)
    agent_config = CustomAgentConfig()
    agent = lr.Agent(agent_config)
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
