from tests.hydra.function import agent_function
from tests.configs import CustomAgentConfig
from llmagent.agent.base import Agent
from llmagent.utils import configuration
from transformers.utils import logging
import hydra
from hydra.core.config_store import ConfigStore

logging.set_verbosity(logging.ERROR)  # for transformers logging


# Register the config with Hydra's ConfigStore
# cs = ConfigStore.instance()
# cs.store(name=CustomAgentConfig.__name__, node=CustomAgentConfig)


@hydra.main(version_base=None, config_name=CustomAgentConfig.__name__)
def main(config: CustomAgentConfig) -> None:
    agent_function(config)


if __name__ == "__main__":
    main()
