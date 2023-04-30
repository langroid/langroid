from tests.configs import CustomAgentConfig
from llmagent.agent.base import Agent
from llmagent.utils import configuration
from transformers.utils import logging

logging.set_verbosity(logging.ERROR)  # for transformers logging


def agent_function(config: CustomAgentConfig):
    configuration.update_global_settings(config, keys=["debug"])
    agent = Agent(config)
    response = agent.respond("what is the capital of France?")  # direct LLM question
    return response
