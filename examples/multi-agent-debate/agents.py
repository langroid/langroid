from langroid.language_models import OpenAIGPTConfig
from langroid import ChatAgent, ChatAgentConfig


def create_agent(base_llm_config: OpenAIGPTConfig, system_message: str) -> ChatAgent:
    """Create a ChatAgent with the specified system message.

    Initializes a `ChatAgent` using the provided LLM configuration and system
    message.

    Args:
        base_llm_config (OpenAIGPTConfig): Configuration for the base LLM.
        system_message (str): The system message to initialize the agent with.

    Returns:
        ChatAgent: A configured `ChatAgent` instance.
    """
    config = ChatAgentConfig(
        llm=base_llm_config,
        system_message=system_message,
        vecdb=None,  # Explicitly disable vector database
    )
    return ChatAgent(config)
