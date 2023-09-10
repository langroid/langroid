"""
The most basic chatbot example, using the default settings.
A single Agent allows you to chat with a pre-trained Language Model.

Run like this:

python3 examples/basic/chat.py

Use optional arguments to change the settings, e.g.:

-ll # use locally running Llama model
-lc 1000 # use local Llama model with context length 1000
-ns # no streaming
-d # debug mode
-nc # no cache
-ct momento # use momento cache (instead of redis)

For details on running with local Llama model, see:
https://langroid.github.io/langroid/tutorials/llama/
"""
import typer
from rich import print
from rich.prompt import Prompt
from pydantic import BaseSettings
from dotenv import load_dotenv

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
    LocalModelConfig,
)
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


class CLIOptions(BaseSettings):
    local_llm: bool = False
    local_ctx: int = 2048

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

    # create the appropriate OpenAIGPTConfig depending on local model or not

    if opts.local_llm:
        # assumes local endpoint is either the default http://localhost:8000/v1
        # or if not, it has been set in the .env file as the value of
        # OPENAI_LOCAL.API_BASE
        llm_config = OpenAIGPTConfig(
            chat_model=OpenAIChatModel.LOCAL,
            local=LocalModelConfig(
                context_length=opts.local_ctx,
            ),
        )
    else:
        # defaults to chat_model = OpenAIChatModel.GPT4
        llm_config = OpenAIGPTConfig()

    default_sys_msg = (
        "You are a helpful assistant. Ask me how you can help. "
        "Be very concise in your answers."
        if llm_config.chat_model == OpenAIChatModel.LOCAL
        else "You are a helpful assistant."
    )
    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default=default_sys_msg,
    )

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=llm_config,
    )
    agent = ChatAgent(config)
    task = Task(agent)
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    local_llm: bool = typer.Option(False, "--local_llm", "-ll", help="use local llm"),
    local_ctx: int = typer.Option(
        2048, "--local_ctx", "-lc", help="local llm context size (default 2048)"
    ),
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
    opts = CLIOptions(
        local_llm=local_llm,
        local_ctx=local_ctx,
    )
    chat(opts)


if __name__ == "__main__":
    app()
