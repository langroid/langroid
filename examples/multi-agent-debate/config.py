from rich.prompt import Prompt
import langroid.language_models as lm
import langroid.utils.configuration
from langroid.language_models import OpenAIGPTConfig
from langroid.utils.configuration import Settings

# Constants
MODEL_MAP = {
    "1": lm.OpenAIChatModel.GPT4o,
    "2": lm.OpenAIChatModel.GPT4,
    "3": lm.OpenAIChatModel.GPT4o_MINI,
    "4": lm.OpenAIChatModel.GPT4_TURBO,
    "5": lm.OpenAIChatModel.GPT4_32K,
    "6": lm.OpenAIChatModel.GPT3_5_TURBO,
    "7": "ollama/mistral:7b-instruct-v0.2-q8_0",
    "8": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH,
    "9": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH_8B,
    "10": "gemini/" + lm.GeminiModel.GEMINI_1_5_PRO,
}
DEFAULT_MAX_OUTPUT_TOKENS = 15_00
MISTRAL_MAX_OUTPUT_TOKENS = 16_000


def get_global_settings(debug: bool = False, nocache: bool = True) -> Settings:
    """
    Retrieve global Langroid settings.

    Args:
        debug (bool): If True, enables debug mode.
        nocache (bool): If True, disables caching.

    Returns:
        Settings: Langroid's global configuration object.
    """
    return langroid.utils.configuration.Settings(
        debug=debug,
        cache=not nocache,
    )


def select_model(config_agent_name: str) -> str:
    """
    Prompt the user to select an OpenAI or Gemini model for the specified agent.

    This function prompts the user to select an option from  a list of available models.
    The user's input corresponds to a predefined choice, which is
    then returned as a string representing the selected option.

    Args:
        config_agent_name (str): The name of the agent being configured, used
                                 in the prompt to personalize the message.

    Returns:
        str: The user's selected option as a string, corresponding to one of the
             predefined model choices (e.g., "1", "2", ..., "10").

    """
    return Prompt.ask(
        f"Select a Model for {config_agent_name}:\n"
        "1: gpt-4o\n"
        "2: gpt-4\n"
        "3: gpt-4o-mini\n"
        "4: gpt-4-turbo\n"
        "5: gpt-4-32k\n"
        "6: gpt-3.5-turbo-1106\n"
        "7: Mistral: mistral:7b-instruct-v0.2-q8_0a\n"
        "8: Gemini: gemini-1.5-flash\n"
        "9: Gemini: gemini-1.5-flash-8b\n"
        "10: Gemini: gemini-1.5-pro\n",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default="1",
    )


def create_llm_config(chat_model_option: str) -> OpenAIGPTConfig:
    """
    Creates an LLM (Language Learning Model) configuration based on the selected model.

    This function uses the user's selection (identified by `chat_model_option`)
    to retrieve the corresponding chat model from the `MODEL_MAP` and create
    an `OpenAIGPTConfig` object with the specified settings.

    Args:
        chat_model_option (str): The key corresponding to the user's selected model.

    Returns:
        OpenAIGPTConfig: A configuration object for the selected LLM.

    Raises:
        ValueError: If the user provided`chat_model_option` does not exist in `MODEL_MAP`.
    """
    chat_model = MODEL_MAP.get(chat_model_option)
    if not chat_model:
        raise ValueError(f"Invalid model selection: {chat_model_option}")

    # Determine max_output_tokens based on the selected model
    max_output_tokens = (
        MISTRAL_MAX_OUTPUT_TOKENS if chat_model_option == "7" else DEFAULT_MAX_OUTPUT_TOKENS
    )

    return lm.OpenAIGPTConfig(
        chat_model=chat_model,
        max_output_tokens=max_output_tokens,
    )


def get_base_llm_config(config_agent_name: str) -> OpenAIGPTConfig:
    """
    Prompt the user to select a base LLM configuration and return it.

    Args:
        config_agent_name (str): The name of the agent being configured.

    Returns:
        OpenAIGPTConfig: The selected LLM's configuration.
    """
    chat_model_option = select_model(config_agent_name)
    return create_llm_config(chat_model_option)
