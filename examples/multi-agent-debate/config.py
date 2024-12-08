from rich.prompt import Prompt
import langroid.language_models as lm
import langroid.utils.configuration


def get_global_settings(debug=False, nocache=True):
    """
    Returns global settings for Langroid.

    Args:
        debug (bool): Enable or disable debug mode.
        nocache (bool): Enable or disable caching.

    Returns:
        Settings: Langroid configuration settings.
    """
    return langroid.utils.configuration.Settings(
        debug=debug,
        cache=not nocache,
    )


def get_base_llm_config():
    """
    Prompts the user to select a base LLM configuration.

    Returns:
        OpenAIGPTConfig: Base configuration for the selected LLM.
    """
    chat_model_option = Prompt.ask(
        "Which OpenAI Model do you want to use? Select an option:\n"
        "1: gpt-4o\n"
        "2: gpt-4\n"
        "3: gpt-4o-mini\n"
        "4: gpt-4-turbo\n"
        "5: gpt-4-32k \n"
        "6: gpt-3.5-turbo-1106 \n"  
        "7: Mistral: mistral:7b-instruct-v0.2-q8_0a\n"
        "8: Gemini:gemini-1.5-flash \n"
        "9: Gemini:gemini-1.5-flash-8b \n"
        "10: Gemini:gemini-1.5-pro \n"
        "Enter 1, 2, 3, 0r 4:",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default="1"
    )

    model_map = {
        "1": lm.OpenAIChatModel.GPT4o,
        "2": lm.OpenAIChatModel.GPT4,
        "3": lm.OpenAIChatModel.GPT4o_MINI,
        "4": lm.OpenAIChatModel.GPT4_TURBO,
        "5": lm.OpenAIChatModel.GPT4_32K,
        "6": lm.OpenAIChatModel.GPT3_5_TURBO,
        "7": "ollama/mistral:7b-instruct-v0.2-q8_0",
        "8": "gemini/gemini-1.5-flash",
        "9": "gemini/gemini-1.5-flash-8b",
        "10": "gemini/gemini-1.5-pro",
    }
    chat_model = model_map[chat_model_option]
    base_llm_config = lm.OpenAIGPTConfig(
        chat_model=chat_model,
        max_output_tokens=1500,  # Adjusted to prevent truncation
    )
    return base_llm_config
