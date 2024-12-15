"""
Demonstrating the utility of Hypothetical Questions (hq) in the context of a Chatbot.

In the following example a DocChatAgent is created and it can both be queried on it's
documents both in a normal way and in a hypothetical way.

The successful execution of this example depends on the generated hypothetical questions,
but with the current default configuration it should be able to generate questions that
are relevant to the given user query `What pasta is good for a hearty mountain meal?`.
The same query does not return any relevant documents in the normal mode, while in the
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
            chunk_size=60,
            overlap=15,
            min_chunk_chars=50,
            n_similar_docs=1,  # for testing purpose to retrieve only very similar docs
        ),
        hypothetical_questions=use_hq,
        num_hypothetical_questions=3,
        hypothetical_questions_prompt="""
        Given the following text passage, generate up to %(num_hypothetical_questions)s diverse and contextually relevant questions that could be asked about this passage. 
        Focus on questions that align with the text's topics, such as regional pasta specialties, preparation methods, historical facts, cultural significance, and specific ingredient pairings. 
        Make the questions varied and avoid repetition.

        PASSAGE:
        %(passage)s

        EXAMPLES OF QUESTIONS:
        - What is the origin of [specific pasta name]?
        - Which region is known for [specific pasta type]?
        - In which natural setting is [specific pasta] typically enjoyed?
        - What are the typical ingredients in [specific dish]?
        - How is [specific pasta] traditionally served or prepared?
        """,
        hypothetical_questions_batch_size=7,
    )

    doc_agent = DocChatAgent(config=config)
    print("[blue]Welcome to the document chatbot!")
    doc_agent.ingest_doc_paths(
        "https://www.italia.it/en/italy/things-to-do/pasta-types-italian-formats-and-recipes"
    )

    doc_task = Task(
        doc_agent,
        interactive=False,
        name="DocAgent",
    )

    print("[cyan]Enter x or q to quit, or ? for evidence")

    doc_task.run("What pasta is good for a hearty mountain meal?")


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
