"""
The most basic chatbot example, using the default settings.
A single Agent allows you to chat with a pre-trained Language Model.

Run like this:

python3 examples/basic/chat.py

Use optional arguments to change the settings, e.g.:

-m "local" to use a model served locally at an OpenAI-API-compatible endpoint
[ Ensure the API endpoint url matches the one in the code below, or edit it. ]
OR
- m "litellm/ollama/llama2" to use any model supported by litellm
(see list here https://docs.litellm.ai/docs/providers)
[Note you must prepend "litellm/" to the model name required in the litellm docs,
e.g. "ollama/llama2" becomes "litellm/ollama/llama2",
"bedrock/anthropic.claude-instant-v1" becomes
"litellm/bedrock/anthropic.claude-instant-v1"]

-ns # no streaming
-d # debug mode
-nc # no cache
-ct momento # use momento cache (instead of redis)

For details on running with local Llama model, see:
https://langroid.github.io/langroid/blog/2023/09/14/using-langroid-with-local-llms/
"""
import typer
from rich import print
from rich.prompt import Prompt
from pydantic import BaseSettings
from dotenv import load_dotenv

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()

# Create classes for other model configs

# Use this config for any model supported by litellm.
# (see list here https://docs.litellm.ai/docs/providers)
LiteLLMOllamaConfig = OpenAIGPTConfig.create(prefix="ollama")
litellm_ollama_config = LiteLLMOllamaConfig(
    chat_model="ollama/llama2",  # or, e.g., "bedrock/anthropic.claude-instant-v1"
    api_base="http://localhost:11434",  # not needed for models at remote APIs
    litellm=True,
    chat_context_length=2048,  # adjust based on model
)

# Use this config for any model that is locally served at an
# OpenAI-compatible API endpoint. In this case the `chat_model` name is ignored,
# but you must set the `api_base` to the URL where the model is listening.

LocalConfig = OpenAIGPTConfig.create(prefix="local")
local_config = LocalConfig(
    chat_model="local",  # doesn't matter
    api_base="http://localhost:8000/v1",  # <- edit if running at a different port
    litellm=False,
    chat_context_length=2048,  # adjust based on model
)


class CLIOptions(BaseSettings):
    model: str = ""

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    if opts.model == "local":
        llm_config = local_config
    elif opts.model.startswith("litellm"):
        # e.g. litellm/ollama/llama2 or litellm/bedrock/anthropic.claude-instant-v1
        llm_config = litellm_ollama_config
        llm_config.chat_model = opts.model.split("/", 1)[1]  # => ollama/llama2

    else:
        llm_config = OpenAIGPTConfig()

    default_sys_msg = "You are a helpful assistant. Be concise in your answers."

    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default=default_sys_msg,
    )

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=llm_config,
        vecdb=None,
    )
    agent = ChatAgent(config)
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
