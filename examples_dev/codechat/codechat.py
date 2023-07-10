import re
import warnings

import typer
from rich import print
from rich.prompt import Prompt

from examples_dev.codechat.code_chat_agent import CodeChatAgent, CodeChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel
from langroid.parsing.urls import get_list_from_user
from langroid.utils import configuration
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()

DEFAULT_URL = "https://github.com/hyperonym/basaran"


def chat(config: CodeChatAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4
    default_urls = [DEFAULT_URL]

    print("[blue]Welcome to the GitHub Repo chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print("[blue]Enter a GitHub URL below (or leave empty for default Repo)")
    urls = get_list_from_user(n=1) or default_urls
    url = urls[0]
    config.repo_url = url

    agent = CodeChatAgent(config)

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )

    system_msg = Prompt.ask(
        """
        [blue] Tell me who I am; complete this sentence: You are...
        """,
        default="a coding expert, who will help me understand a code repo",
    )

    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)

    task = Task(
        agent,
        llm_delegate=False,
        single_round=False,
        system_message="You are " + system_msg,
    )
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="do no use cache"),
) -> None:
    config = CodeChatAgentConfig(debug=debug, gpt4=True, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
