import logging
from langroid.language_models import OpenAIGPTConfig
from langroid import ChatAgent, ChatAgentConfig

logger = logging.getLogger(__name__)


def create_agent(base_llm_config: OpenAIGPTConfig, system_message: str) -> ChatAgent:
    """Creates a ChatAgent with a given system message."""
    config = ChatAgentConfig(
        llm=base_llm_config,
        system_message=system_message,
        # Removed 'system_message_role' and 'verbose' parameters
    )
    return ChatAgent(config)
