from llmagent.parsing.urls import get_list_from_user, get_urls_and_paths
from llmagent.utils.logging import setup_colored_logging
from examples.urlqa.config import URLQAConfig
from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.language_models.openai_gpt import OpenAIChatModel
from llmagent.mytypes import Document
from llmagent.parsing.url_loader import URLLoader
from rich.prompt import Prompt
from llmagent.parsing.repo_loader import RepoLoader
from llmagent.utils import configuration
from typing import List
import re
import typer


import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


def chat(config: URLQAConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4

    default_urls = config.urls

    print("[blue]Welcome to the document chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print(
        "[blue]Enter some URLs or file/dir paths below "
        " (or leave empty for default URLs)"
    )
    inputs = get_list_from_user() or default_urls
    urls, paths = get_urls_and_paths(inputs)

    collection_name = Prompt.ask(
        "What would you like to name this collection?",
        default=config.vecdb.collection_name,
    )
    config.vecdb.collection_name = collection_name
    documents = []
    if len(urls) > 0:
        loader = URLLoader(urls=urls)
        documents: List[Document] = loader.load()
    if len(paths) > 0:
        for p in paths:
            path_docs = RepoLoader.get_documents(p)
            documents.extend(path_docs)

    agent = DocChatAgent(config)
    nsplits = agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(
        f"""
    [green]I have processed the following {len(urls)} URLs 
    and {len(paths)} paths into {nsplits} parts:
    """.strip()
    )
    print("\n".join(urls))
    print("\n".join(paths))

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )
    system_msg = Prompt.ask(
        """
    [blue] Tell me who I am; complete this sentence: You are...
    [or hit enter for default] 
    [blue] Human
    """,
        default="a helpful assistant.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)
    agent.run(system_message="You are " + system_msg)


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
