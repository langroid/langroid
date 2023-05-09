from llmagent.agent.base import Agent, AgentConfig
from llmagent.language_models.base import LLMConfig, StreamingIfAllowed
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.system import rmdir


def test_agent():
    """
    Test whether the combined configs work as expected.
    """
    qd_dir = ".qdrant/testdata_test_agent"
    rmdir(qd_dir)
    cfg = AgentConfig(
        name="test-llmagent",
        debug=False,
        vecdb=None,
        llm=LLMConfig(
            type="openai",
        ),
        parsing=None,
        prompts=PromptsConfig(),
    )

    agent = Agent(cfg)
    response = agent.respond("what is the capital of France?")  # direct LLM question
    assert "Paris" in response.content

    with StreamingIfAllowed(agent.llm, False):
        response = agent.respond("what is the capital of France?")
    assert "Paris" in response.content
    rmdir(qd_dir)
