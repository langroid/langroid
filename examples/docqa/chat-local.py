"""
Single agent to use to chat with an LLM using  Retrieval-Augmented Generation (RAG).
Similar to chat.py but allows specifying a local LLM.

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

NOTES:
(1) The app works best with GPT4/Turbo, but results may be mixed with local LLMs.
You may have to tweak the system_message, use_message, and summarize_prompt
as indicated in comments below, to get good results.
(2) The default vector-db in DocChatAgent is QdrantDB, but you can switch to the
other supported vector-dbs, e.g. lancedb or chroma.

"""

import os
import re

import typer
from rich import print
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.agent.task import Task
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=32_000,  # adjust based on model
        timeout=90,
    )

    config = DocChatAgentConfig(
        n_query_rephrases=0,
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
        n_similar_chunks=3,
        n_relevant_chunks=3,
        llm=llm_config,
        # relevance_extractor_config=None,
        # system_message="...override default DocChatAgent system msg here",
        # user_message="...override default DocChatAgent user msg here",
        # summarize_prompt="...override default DocChatAgent summarize prompt here",
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=300,  # aim for this many tokens per chunk
            overlap=30,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=5,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "unstructured", "docling", "fitz"
                library="pymupdf4llm",
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
        system_message="You are " + system_msg,
    )
    task.run()


if __name__ == "__main__":
    app()
