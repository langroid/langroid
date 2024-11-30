from rich.prompt import Prompt
import langroid.language_models as lm
import langroid.utils.configuration


# Define the streaming handler function here
def handle_streaming_output(token: str):
    print(token, end='', flush=True)


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


def get_base_llm_config(streamer=None):
    """
    Prompts the user to select a base LLM configuration.

    Args:
        streamer (Callable): Function to handle streaming tokens.

    Returns:
        OpenAIGPTConfig: Base configuration for the selected LLM.
    """
    chat_model_option = Prompt.ask(
        "Which OpenAI Model do you want to use? Select an option:\n"
        "1: GPT4o\n"
        "2: GPT4\n"
        "3: Mistral: mistral:7b-instruct-v0.2-q8_0a\n"
        "Enter 1, 2, or 3:",
        choices=["1", "2", "3"],
        default="1"
    )

    model_map = {
        "1": "gpt-4o",
        "2": "gpt-4",
    }

    if chat_model_option == "3":
        chat_model = "ollama/mistral:7b-instruct-v0.2-q8_0"
        base_llm_config = lm.OpenAIGPTConfig(
            chat_model=chat_model,
            chat_context_length=16000,  # Only set for Ollama model
            max_output_tokens=1500,  # Adjusted to prevent truncation
            stream=True,  # Enable streaming outputs
            streamer=streamer,  # Set the streaming handler
        )
    else:
        chat_model = model_map[chat_model_option]
        base_llm_config = lm.OpenAIGPTConfig(
            chat_model=chat_model,
            max_output_tokens=1500,  # Adjusted to prevent truncation
            stream=True,  # Enable streaming outputs
            streamer=streamer,  # Set the streaming handler
        )
    return base_llm_config
