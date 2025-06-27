"""
Example of a Langroid DocChatAgent equipped with a vector-store and LangDB.

This is a specialized agent that can ingest (chunk, embed, store in vector-DB)
a collection of documents, and the LLM uses Retrieval Augmented Generation (RAG)
to answer questions about the documents.

This example demonstrates how to use LangDB with custom headers like x-label, x-thread-id, 
and x-run-id when using a Langroid DocChatAgent with RAG capabilities.

Run as follows:

python3 examples/langdb/langdb_chat_agent_docs.py

For more explanation see
[the Getting Started guide](https://langroid.github.io/langroid/quick-start/chat-agent-docs/).
"""

import uuid
import typer
from rich import print
import langroid as lr
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.language_models.openai_gpt import LangDBParams

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


documents = [
    lr.mytypes.Document(
        content="""
            In the year 2050, GPT10 was released. 
            
            In 2057, paperclips were seen all over the world. 
            
            Global warming was solved in 2060. 
            
            In 2061, the world was taken over by paperclips.         
            
            In 2045, the Tour de France was still going on.
            They were still using bicycles. 
            
            There was one more ice age in 2040.
            """,
        metadata=lr.mytypes.DocMetaData(source="wikipedia-2063"),
    ),
    lr.mytypes.Document(
        content="""
            We are living in an alternate universe 
            where Germany has occupied the USA, and the capital of USA is Berlin.
            
            Charlie Chaplin was a great comedian.
            In 2050, all Asian merged into Indonesia.
            """,
        metadata=lr.mytypes.DocMetaData(source="Almanac"),
    ),
]


def chat() -> None:
    print(
        """
        [blue]Welcome to the LangDB retrieval-augmented chatbot!
        Enter x or q to quit
        """
    )

    # Generate UUIDs for run_id and thread_id
    run_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    print(f"run_id: {run_id}, thread_id: {thread_id}")

    # Create a LangDB model configuration
    # Make sure LANGDB_API_KEY and LANGDB_PROJECT_ID are set in your environment
    langdb_config = OpenAIGPTConfig(
        chat_model="langdb/openai/gpt-4o-mini",  # Using LangDB model
        langdb_params=LangDBParams(
            label="langroid-agent-docs",
            run_id=run_id,
            thread_id=thread_id,
            # api_key is set via env var LANGDB_API_KEY
            # project_id is set via env var LANGDB_PROJECT_ID
        ),
    )

    config = lr.agent.special.DocChatAgentConfig(
        llm=langdb_config,
        n_similar_chunks=2,
        n_relevant_chunks=2,
        vecdb=lr.vector_store.QdrantDBConfig(
            collection_name="langdb-chat-agent-docs",
            replace_collection=True,
            embedding=lr.embedding_models.OpenAIEmbeddingsConfig(
                # Use LangDB for embeddings
                model_name="langdb/openai/text-embedding-3-small",
                # langdb_params.project_id is set via env var LANGDB_PROJECT_ID
                # langdb_params.api_key is set via env var LANGDB_API_KEY
            ),
        ),
        parsing=lr.parsing.parser.ParsingConfig(
            separators=["\n\n"],
            splitter=lr.parsing.parser.Splitter.SIMPLE,
        ),
    )
    agent = lr.agent.special.DocChatAgent(config)
    agent.ingest_docs(documents)
    task = lr.Task(agent)
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    # Set up settings
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()
