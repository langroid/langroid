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
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chat(config: DocChatAgentConfig) -> None:
    agent = DocChatAgent(config)
    print("[blue]Welcome to the document chatbot!")
    agent.user_docs_ingest_dialog()
    print("[cyan]Enter x or q to quit, or ? for evidence")

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
        hypothetical_answer=False,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=1000,  # aim for this many tokens per chunk
            overlap=100,  # overlap between chunks
            max_chunks=10_000,
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=3,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "haystack", "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
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
