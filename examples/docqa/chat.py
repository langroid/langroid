"""
Single agent to use to chat with a Retrieval-augmented LLM.
Repeat: User asks question -> LLM answers.

Run like this:
python3 examples/docqa/chat.py

"""
import typer
from rich import print
import os

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)

from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=16_000,  # adjust as needed
        temperature=0.2,
        max_output_tokens=300,
    )

    config = DocChatAgentConfig(
        llm=llm_config,
        n_query_rephrases=0,
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=10,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=200,  # aim for this many tokens per chunk
            overlap=50,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=25,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=3,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
    )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    agent = DocChatAgent(config)
    print("[blue]Welcome to the document chatbot!")
    agent.user_docs_ingest_dialog()
    print("[cyan]Enter x or q to quit, or ? for evidence")

    task = lr.Task(
        agent,
        system_message="You are a helpful assistant, "
        "answering questions about some docs",
    )
    task.run()


if __name__ == "__main__":
    app()
