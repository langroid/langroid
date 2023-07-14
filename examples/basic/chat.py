import os
import textwrap
import warnings

import typer
from rich import print

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.vector_store.base import VectorStoreConfig

app = typer.Typer()

setup_colored_logging()


class BasicConfig(ChatAgentConfig):
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4?
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: None | VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
    )


def chat(config: BasicConfig) -> None:
    print(
        textwrap.dedent(
            """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit
        """
        ).strip()
    )

    agent = ChatAgent(config)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )
    task = Task(agent)
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    config = BasicConfig()
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat(config)


if __name__ == "__main__":
    app()
