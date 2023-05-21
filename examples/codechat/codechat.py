from llmagent.parsing.urls import get_urls_from_user
from llmagent.utils.logging import setup_colored_logging
from llmagent.utils import configuration
from llmagent.language_models.openai_gpt import OpenAIChatModel
from examples.codechat.code_chat_agent import CodeChatAgent, CodeChatAgentConfig

import typer
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()




def chat(config: CodeChatAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4
    default_urls = [config.repo_url]

    print("[blue]Welcome to the GitHub Repo chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print("[blue]Enter a GitHub URL below (or leave empty for default Repo)")
    urls = get_urls_from_user(n=1) or default_urls
    config.repo_url = urls[0]
    agent = CodeChatAgent(config)

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
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use GPT-4"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="do no use cache"),
) -> None:
    config = CodeChatAgentConfig(debug=debug, gpt4=gpt4, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
