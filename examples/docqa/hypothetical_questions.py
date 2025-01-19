"""
Demonstrating the utility of Hypothetical Questions (HQ) in the context of a
DocChatAgent.

In the following example, a DocChatAgent is created and it can be queried on its
documents both in a normal way and in a hypothetical way.

Although this is being referred to as Hypothetical Questions, it is not limited to
just questions -- it is simply a way to augment the document-chunks at ingestion time,
with keywords that increase the "semantic surface" of the chunks to improve
retrieval accuracy.

This example illustrates the benefit of HQ in a medical scenario
where each "document chunk" is simply the name of a medical test
(e.g. "cholesterol", "BUN", "PSA", etc)
and when `use_hypothetical_question` is enabled,
the chunk (i.e. test name) is augment it with keywords that add more
context, such as which organ it is related to
(e.g., "heart", "kidney", "prostate", etc).
This way, when a user asks "which tests are related to kidney health",
these augmentations ensure that the test names are retrieved more accurately.

Running the script compares the accuracy of
results of the DocChatAgent with and without HQ.

Run like this to use HQ:

python3 examples/docqa/hypothetical_questions.py

or without HQ:

python3 examples/docqa/hypothetical_questions.py --no-use-hq
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
