from llmagent.parsing.urls import get_urls_from_user
from llmagent.utils.logging import setup_colored_logging
from examples.urlqa.config import URLQAConfig
from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.mytypes import Document
from llmagent.parsing.url_loader import URLLoader
from llmagent.utils import configuration
from typing import List
import typer


import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


def chat(config: URLQAConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream"])

    default_urls = config.urls

    print("[blue]Welcome to the URL chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print("[blue]Enter some URLs below (or leave empty for default URLs)")
    urls = get_urls_from_user() or default_urls
    loader = URLLoader(urls=urls)
    documents: List[Document] = loader.load()

    agent = DocChatAgent(config)
    nsplits = agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(
        f"""
    [green]I have processed the following {len(urls)} URLs into {nsplits} parts:
    """.strip()
    )
    print("\n".join(urls))

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )

    while True:
        print("\n[blue]Query: ", end="")
        query = input("")
        if query in ["exit", "quit", "q", "x", "bye"]:
            print("[green] Bye, hope this was useful!")
            break
        agent.respond(query)


@app.command()
def main(debug: bool = typer.Option(False, "--debug", "-d", help="debug mode")) -> None:
    config = URLQAConfig(debug=debug)
    chat(config)


if __name__ == "__main__":
    app()
