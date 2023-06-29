from llmagent.parsing.urls import get_list_from_user, get_urls_and_paths
from llmagent.utils.logging import setup_colored_logging
from llmagent.agent.task import Task
from examples.urlqa.config import URLQAConfig
from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.language_models.openai_gpt import OpenAIChatModel
from llmagent.mytypes import Document
from llmagent.parsing.url_loader import URLLoader
from llmagent.parsing.repo_loader import RepoLoader
from llmagent.utils import configuration
from typing import List
import re
import typer


import os
from rich import print
from rich.prompt import Prompt
import warnings

app = typer.Typer()

setup_colored_logging()


def chat(config: URLQAConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4

    default_urls = config.urls
    agent = DocChatAgent(config)
    n_deletes = agent.vecdb.clear_empty_collections()
    collections = agent.vecdb.list_collections()
    collection_name = "NEW"
    is_new_collection = False
    if len(collections) > 1:
        n = len(collections)
        delete_str = f"(deleted {n_deletes} empty collections)" if n_deletes > 0 else ""
        print(f"Found {n} collections: {delete_str}")
        for i, option in enumerate(collections, start=1):
            print(f"{i}. {option}")
        while True:
            choice = Prompt.ask(
                f"Enter a number in the range [1, {n}] to select a collection, "
                "or hit enter to create a NEW collection",
                default="0",
            )
            if choice.isdigit() and 0 <= int(choice) <= n:
                break

        if int(choice) > 0:
            collection_name = collections[int(choice) - 1]
            print(f"Using collection {collection_name}")

    if collection_name == "NEW":
        is_new_collection = True
        collection_name = Prompt.ask(
            "What would you like to name the NEW collection?", default="urlqa-chat"
        )
    config.vecdb.collection_name = collection_name

    agent.vecdb.set_collection(collection_name)

    print("[blue]Welcome to the document chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    default_urls_str = " (or leave empty for default URLs)" if is_new_collection else ""
    print("[blue]Enter some URLs or file/dir paths below " f"{default_urls_str}")
    inputs = get_list_from_user()
    if len(inputs) == 0 and is_new_collection:
        inputs = default_urls
    urls, paths = get_urls_and_paths(inputs)
    documents = []
    if len(urls) > 0:
        loader = URLLoader(urls=urls)
        documents: List[Document] = loader.load()
    if len(paths) > 0:
        for p in paths:
            path_docs = RepoLoader.get_documents(p)
            documents.extend(path_docs)

    if len(documents) > 0:
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
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use gpt4"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    config = URLQAConfig(debug=debug, gpt4=gpt4, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
