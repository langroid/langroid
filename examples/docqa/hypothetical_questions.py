"""
Demonstrating the utility of Hypothetical Questions (hq) in the context of a Chatbot.

In the following example, a DocChatAgent is created and it can be queried on its
documents both in a normal way and in a hypothetical way.

The successful execution of this example depends on the generated hypothetical questions,
but with the current default configuration, it should be able to generate questions that
are relevant to the given user query `How does the system generate responses in conversation mode?`.
The same query does not return accurate answers in the normal mode, while in the
hypothetical mode it returns a relevant document.

Run like this:

python3 examples/docqa/hypothetical_questions.py --no-debug --reset --no-cache --docker --use-hq
"""

import typer
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.parsing.parser import ParsingConfig
from langroid.utils.configuration import Settings
from langroid.vector_store.qdrantdb import QdrantDBConfig
from langroid.utils.constants import NO_ANSWER

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


def setup_vecdb(docker: bool, reset: bool, collection: str) -> QdrantDBConfig:
    """Configure vector database."""
    return QdrantDBConfig(
        collection_name=collection, replace_collection=reset, docker=docker
    )


def run_document_chatbot(
    model: str,
    docker: bool,
    reset: bool,
    collection: str,
    use_hq: bool,
) -> None:
    """
    Main function for the document chatbot.

    Args:
        model: chat model
        docker: use docker for vector database
        reset: reset conversation memory
        collection: collection name
        use_hq: use hypothetical
    """
    llm_config = lm.OpenAIGPTConfig(chat_model=model)
    vecdb_config = setup_vecdb(docker=docker, reset=reset, collection=collection)
    config = DocChatAgentConfig(
        llm=llm_config,
        vecdb=vecdb_config,
        hypothetical_answer=False,
        parsing=ParsingConfig(
            chunk_size=120,
            overlap=15,
            min_chunk_chars=50,
        ),
        # n_neighbor_chunks=1,
        num_hypothetical_questions=3 if use_hq else 0,
        hypothetical_questions_prompt="""
        You are a python expert professor and you are analyzing a python script to generate helpful insights
        that your students might wonder. For each segment, consider:
        Given the following python script, generate up to %(num_hypothetical_questions)s diverse 
        and specific quotes explaining the code.
        Ensure the quotes reflect different angles or use cases related to the text.
        Only write the quotes, no intro needed.

        PASSAGE:
        %(passage)s
        """,
        hypothetical_questions_batch_size=30,
    )

    doc_agent = DocChatAgent(config=config)
    doc_agent.ingest_doc_paths(
        "langroid/agent/special/doc_chat_agent.py",
    )

    doc_task = Task(
        doc_agent,
        interactive=False,
        name="DocAgent",
        single_round=True,
    )

    user_query = "How does the system generate responses in conversation mode?"
    print(f"[blue]Welcome to the document chatbot!..answering to: '{user_query}'")
    print("[cyan]Enter x or q to quit, or ? for evidence")

    res = doc_task.run(user_query)

    if not res or res.content == NO_ANSWER:
        print("[red]No answer found")
        return

    print(f"[green]Answer: {res.content}")


@app.command()
def main(
    debug: bool = typer.Option(
        False, "--debug/--no-debug", "-d", help="Enable debug mode"
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", "-s", help="Enable streaming output"
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", "-c", help="Enable caching"),
    model: str = typer.Option(
        lm.OpenAIChatModel.GPT4o_MINI.value, "--model", "-m", help="Chat model to use"
    ),
    collection: str = typer.Option(
        "docchat_hq", "--collection", help="Collection name for vector database"
    ),
    docker: bool = typer.Option(
        True, "--docker/--no-docker", help="Use docker for vector database"
    ),
    reset: bool = typer.Option(
        True, "--reset/--no-reset", help="Reset conversation memory"
    ),
    use_hq: bool = typer.Option(
        True, "--use-hq/--no-use-hq", help="Use hypothetical questions"
    ),
) -> None:
    """Main app function."""
    lr.utils.configuration.set_global(
        Settings(
            debug=debug,
            cache=cache,
            stream=stream,
        )
    )

    run_document_chatbot(
        model=model,
        docker=docker,
        collection=collection,
        reset=reset,
        use_hq=use_hq,
    )


if __name__ == "__main__":
    app()
