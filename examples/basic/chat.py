from llmagent.utils.logging import setup_colored_logging
from examples.urlqa.config import URLQAConfig
from llmagent.utils import configuration
from llmagent.agent.base import AgentConfig
from llmagent.agent.chat_agent import ChatAgent
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel

import typer


import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


class BasicConfig(AgentConfig):
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4?
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
    )


def chat(config: BasicConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4

    print(
        """
    [blue]Welcome to the basic chatbot!
    Enter x or q to quit
    """
    )

    agent = ChatAgent(config)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )
    agent.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use gpt4"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    config = URLQAConfig(debug=debug, gpt4=gpt4, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
