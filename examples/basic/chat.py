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
from pydantic import BaseSettings

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from langroid.io.cmd_io import CmdInputProvider, CmdOutputProvider

from langroid.io.base import IOFactory


app = typer.Typer()

setup_colored_logging()

class CLIOptions(BaseSettings):
    local_llm: bool = False
    local_ctx: int = 2048

    class Config:
        extra = "forbid"
        env_prefix = ""
       
def chat() -> None:
    IOFactory.set_provider(CmdInputProvider("input"))
    IOFactory.set_provider(CmdOutputProvider("output"))

    io_input = IOFactory.get_provider("input")
    io_output = IOFactory.get_provider("output")

    io_output(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )

    sys_msg = io_input(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default="You are a helpful assistant",
    )

    api_base = "http://localhost:8000/v1" if opts.local_llm else None
    chat_model = OpenAIChatModel.GPT4 if not opts.local_llm else OpenAIChatModel.LOCAL
    llm_config = OpenAIGPTConfig(
        chat_model=chat_model,
        api_base=api_base,
    )
    if opts.local_ctx != 2048:
        llm_config.context_length = {
            OpenAIChatModel.LOCAL: opts.local_ctx,
        }
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
