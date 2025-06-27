"""
Example showing how to use custom headers with LangDB models.

This example demonstrates how to set custom headers like x-label, x-thread-id, and x-run-id
when using LangDB. These headers are specific to LangDB and won't have any effect with other providers.
"""

from uuid import uuid4

from langroid.language_models.openai_gpt import LangDBParams, OpenAIGPT, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

# Set up settings
settings = Settings(debug=True)
set_global(settings)


def main():
    run_id = str(uuid4())
    thread_id = str(uuid4())

    print(f"run_id: {run_id}, thread_id: {thread_id}")
    # Create a LangDB model configuration
    # Make sure LANGDB_API_KEY and LANGDB_PROJECT_ID are set in your environment
    langdb_config = OpenAIGPTConfig(
        chat_model="langdb/openai/gpt-4o-mini",
        langdb_params=LangDBParams(
            label="langroid", run_id=run_id, thread_id=thread_id
        ),
    )

    print(f"Using model: {langdb_config.chat_model}")

    # Create the model
    langdb_model = OpenAIGPT(langdb_config)

    # Use the model
    response = langdb_model.chat(
        messages="Tell me a short joke about programming", max_tokens=100
    )

    print(f"Response: {response.message}")


if __name__ == "__main__":
    main()
