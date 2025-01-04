from typing import Optional
import langroid.language_models as lm
import langroid.utils.configuration
from langroid.language_models import OpenAIGPTConfig
from langroid.utils.configuration import Settings
from generation_config_models import load_generation_config, GenerationConfig

# Constants
MODEL_MAP = {
    "1": lm.OpenAIChatModel.GPT4o,
    "2": lm.OpenAIChatModel.GPT4,
    "3": lm.OpenAIChatModel.GPT4o_MINI,
    "4": lm.OpenAIChatModel.GPT4_TURBO,
    "5": lm.OpenAIChatModel.GPT4_32K,
    "6": lm.OpenAIChatModel.GPT3_5_TURBO,
    "7": "ollama/mistral:7b-instruct-v0.2-q8_0",
    "8": "gemini/" + lm.GeminiModel.GEMINI_2_FLASH,
    "9": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH,
    "10": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH_8B,
    "11": "gemini/" + lm.GeminiModel.GEMINI_1_5_PRO,
}

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


def create_llm_config(
    chat_model_option: str, temperature: Optional[float] = None
) -> OpenAIGPTConfig:
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
    # Load generation configuration from JSON
    generation_config: GenerationConfig = load_generation_config(
        "examples/multi-agent-debate/generation_config.json"
    )

    if not chat_model:
        raise ValueError(f"Invalid model selection: {chat_model_option}")

    # Determine max_output_tokens based on the selected model
    max_output_tokens_config = (
        MISTRAL_MAX_OUTPUT_TOKENS
        if chat_model_option == "7"
        else generation_config.max_output_tokens
    )

    # Use passed temperature if provided; otherwise, use the one from the JSON config
    effective_temperature = (
        temperature if temperature is not None else generation_config.temperature
    )

    # Create and return the LLM configuration
    return OpenAIGPTConfig(
        chat_model=chat_model,
        min_output_tokens=generation_config.min_output_tokens,
        max_output_tokens=max_output_tokens_config,
        temperature=effective_temperature,
        seed=generation_config.seed,
    )


def get_base_llm_config(
    chat_model_option: str, temperature: Optional[float] = None
) -> OpenAIGPTConfig:
    """
    Prompt the user to select a base LLM configuration and return it.

    Args:
        config_agent_name (str): The name of the agent being configured.

    Returns:
        OpenAIGPTConfig: The selected LLM's configuration.
    """

    # Pass temperature only if it is provided
    if temperature is not None:
        return create_llm_config(chat_model_option, temperature)
    return create_llm_config(chat_model_option)
