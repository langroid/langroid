from llmagent.agent.base import Agent
from tests.configs import CustomAgentConfig


def test_agent():
    """
    Test whether the combined configs work as expected.
    """
    agent_config = CustomAgentConfig()
    agent = Agent(agent_config)
    response = agent.respond("what is the capital of France?")  # direct LLM question
    assert "Paris" in response.content
