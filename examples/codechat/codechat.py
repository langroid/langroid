from llmagent.parsing.urls import get_list_from_user, org_user_from_github
from llmagent.utils.logging import setup_colored_logging
from llmagent.utils import configuration
from llmagent.language_models.openai_gpt import OpenAIChatModel
from examples.codechat.code_chat_agent import CodeChatAgent, CodeChatAgentConfig
import re
import typer
from rich import print
from rich.prompt import Prompt
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
    urls = get_list_from_user(n=1) or default_urls
    url = urls[0]
    config.repo_url = url

    default_collection_name = org_user_from_github(url)
    collection_name = Prompt.ask(
        f"""Creating a vector-store for contents of {url}.
            IMPORTANT: we need a unique collection name for this repo.
            What would you like to name this collection?""",
        default=default_collection_name,
    )
    config.vecdb.collection_name = collection_name


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

    agent.do_task(system_message="You are " + system_msg)


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
