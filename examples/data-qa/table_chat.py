"""
Example showing how to chat with a tabular dataset
"""
import typer
from rich.prompt import Prompt
from rich import print

from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


def chat() -> None:
    print("[blue]Welcome to the tabular-data chatbot!\n")
    path = Prompt.ask("[blue]Enter a local path or URL to a tabular dataset")

    agent = TableChatAgent(
        config=TableChatAgentConfig(
            data=path,
            use_tools=True,
            use_functions_api=False,
            llm=OpenAIGPTConfig(
                chat_model=OpenAIChatModel.GPT4,
            ),
        )
    )
    task = Task(agent)
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
