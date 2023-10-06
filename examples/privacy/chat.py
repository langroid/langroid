"""
Meant to be used with local LLMs, using the -l option (see below).

You type a sentence containing potentially sensitive information, and the agent
will annotate sensitive portions of the sentence with the appropriate category.
You can configure PrivacyAnnotator to recognize only specific sensitive
categories, currently defaults to: ["Medical", "CreditCard", "SSN", "Name"]

Example input:
    "John is 45 years old, lives in Ohio, makes 45K a year, and has diabetes."

Example output:
    "[Name: John] is 45 years old, lives in Ohio, makes 45K a year,
        and has [Medical: diabetes]."

Run like this:

python3 examples/privacy/chat.py

Use optional arguments to change the settings, e.g.:

-l # use locally running Llama model
-lc 1000 # use local Llama model with context length 1000
    (you typically can set ctx len when spinning up local llm API server)
-ns # no streaming
-d # debug mode
-nc # no cache
-ct momento # use momento cache (instead of redis)

For details on running with local Llama model, see:
https://langroid.github.io/langroid/blob/posts/local-llm/
"""
import typer
from rich import print
from pydantic import BaseSettings
from dotenv import load_dotenv

from examples.privacy.privacy_annotator import PrivacyAnnotator, PrivacyAnnotatorConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()

# create classes for other model configs
LiteLLMOllamaConfig = OpenAIGPTConfig.create(prefix="ollama")
litellm_ollama_config = LiteLLMOllamaConfig(
    chat_model="ollama/llama2",
    completion_model="ollama/llama2",
    api_base="http://localhost:11434",
    litellm=True,
    chat_context_length=4096,
    use_completion_for_chat=False,
)
OobaConfig = OpenAIGPTConfig.create(prefix="ooba")
ooba_config = OobaConfig(
    chat_model="local",  # doesn't matter
    completion_model="local",  # doesn't matter
    api_base="http://localhost:8000/v1",  # <- edit if running at a different port
    chat_context_length=2048,
    litellm=False,
    use_completion_for_chat=False,
)


class CLIOptions(BaseSettings):
    model: str = ""

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the privacy mask chatbot!
        Enter any text and I will annotate it with sensitive categories and values.
        """
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    if opts.model == "ooba":
        llm_config = ooba_config
    elif opts.model.startswith("ollama"):
        llm_config = litellm_ollama_config
        llm_config.chat_model = opts.model
    else:
        llm_config = OpenAIGPTConfig()

    config = PrivacyAnnotatorConfig(
        llm=llm_config,
        vecdb=None,
    )
    agent = PrivacyAnnotator(config)
    task = Task(agent)
    # local (llama2) models do not like the first message to be empty
    user_message = "Hello." if (opts.model != "") else None
    task.run(user_message)


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    opts = CLIOptions(model=model)
    chat(opts)


if __name__ == "__main__":
    app()
