from rich.prompt import Prompt, Confirm
import langroid.language_models as lm
import langroid.utils.configuration

def get_global_settings(debug=False, nocache=False):
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
        "1: GPT4o\n"
        "2: GPT4\n"
        "3: GPT3.5\n"
        "4: Mistral: mistral:7b-instruct-v0.2-q8_0a\n"
        "Enter 1, 2, 3, or 4:",
        choices=["1", "2", "3", "4"],
        default="1"
    )

    model_map = {
        "1": "GPT4o",
        "2": "GPT4",
        "3": "GPT3.5"
    }

    if chat_model_option == "4":
        chat_model = "ollama/mistral:7b-instruct-v0.2-q8_0"
        base_llm_config = lm.OpenAIGPTConfig(
            chat_model=chat_model,
            chat_context_length=16_000  # Only set for Ollama model
        )
    else:
        chat_model = getattr(lm.OpenAIChatModel, model_map[chat_model_option])
        base_llm_config = lm.OpenAIGPTConfig(
            chat_model=chat_model
        )

    return base_llm_config


def is_llm_delegate():
    """
    Prompts the user to decide whether the LLM should autonomously continue the debate.

    Returns:
        bool: True if the LLM should operate autonomously, False otherwise.
    """
    llm_delegate_setting = Prompt.ask(
        "Would you like the LLM to autonomously continue the debate without waiting for user input? (True/False)",
        choices=["True", "False"],
        default="False",
    )
    return llm_delegate_setting.lower() == "true"

