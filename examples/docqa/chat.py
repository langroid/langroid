"""
Single agent to use to chat with a Retrieval-augmented LLM.
Repeat: User asks question -> LLM answers.
"""
import re
import typer
from rich import print
from rich.prompt import Prompt
import os

from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.parsing.urls import get_list_from_user
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chat(config: DocChatAgentConfig) -> None:
    agent = DocChatAgent(config)
    n_deletes = agent.vecdb.clear_empty_collections()
    collections = agent.vecdb.list_collections()
    collection_name = "NEW"
    is_new_collection = False
    replace_collection = False
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
            choice = Prompt.ask(
                "Would you like to replace this collection?",
                choices=["y", "n"],
                default="n",
            )
            replace_collection = choice == "y"

    if collection_name == "NEW":
        is_new_collection = True
        collection_name = Prompt.ask(
            "What would you like to name the NEW collection?",
            default="urlqa-chat",
        )

    agent.vecdb.set_collection(collection_name, replace=replace_collection)

    print("[blue]Welcome to the document chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    default_urls_str = " (or leave empty for default URLs)" if is_new_collection else ""
    print(f"[blue]Enter some URLs or file/dir paths below {default_urls_str}")
    inputs = get_list_from_user()
    if len(inputs) == 0:
        if is_new_collection:
            inputs = config.default_paths
    agent.config.doc_paths = inputs
    agent.ingest()
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
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    config = DocChatAgentConfig(
        n_query_rephrases=0,
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            cache_type=cache_type,
        )
    )
    chat(config)


if __name__ == "__main__":
    app()
