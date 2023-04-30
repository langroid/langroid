import pytest
from tests.configs import CustomAgentConfig
from hydra import compose, initialize
from tests.hydra.function import agent_function
from llmagent.mytypes import Document


@pytest.fixture(scope="module")
def hydra_config():
    with initialize(version_base=None):
        cfg = compose(config_name="tests.configs.config")
    return cfg


def test_agent(hydra_config: CustomAgentConfig):
    response: Document = agent_function(hydra_config)
    assert "Paris" in response.content
