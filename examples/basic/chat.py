"""
The most basic chatbot example, using the default settings.
A single Agent allows you to chat with a pre-trained Language Model.
"""
import typer
from rich import print
from rich.prompt import Prompt

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from langroid.io.cmd_io import CmdInputProvider, CmdOutputProvider

from langroid.io.base import IOFactory


app = typer.Typer()

setup_colored_logging()


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
        default="Default: 'You are a helpful assistant'",
    )

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
    )
    agent = ChatAgent(config)
    task = Task(
        agent,
        system_message="""
        You are a helpful assistant. Be very concise in your responses, use no more
        than 1-2 sentences.
        """,
    )
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
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
    chat()


if __name__ == "__main__":
    app()
