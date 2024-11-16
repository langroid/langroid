"""
Single agent to use to chat with a Retrieval-augmented LLM.
Repeat: User asks question -> LLM answers.

Run like this:
python3 examples/docqa/chat.py

To change the model, use the --model flag, e.g.:

python3 examples/docqa/chat.py --model ollama/mistral:7b-instruct-v0.2-q8_0

To change the embedding model, use the --embed flag, e.g.:

python3 examples/docqa/chat.py --embed BAAI/bge-large-en-v1.5

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

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
    vecdb: str = typer.Option(
        "qdrant", "--vecdb", "-v", help="vector db name (default: qdrant)"
    ),
    nostream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    embed: str = typer.Option(
        "openai",
        "--embed",
        "-e",
        help="sentence transformer model name",
        # e.g. NeuML/pubmedbert-base-embeddings
    ),
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=16_000,  # adjust as needed
        temperature=0.2,
        max_output_tokens=300,
        timeout=60,
    )

    config = DocChatAgentConfig(
        llm=llm_config,
        n_query_rephrases=0,
        hypothetical_answer=False,
        # how many sentences in each segment, for relevance-extraction:
        # increase this if you find that relevance extraction is losing context
        extraction_granularity=3,
        # for relevance extraction
        # relevance_extractor_config=None,  # set to None to disable relevance extraction
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=2,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=200,  # aim for this many tokens per chunk
            overlap=50,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=5,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=5,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "unstructured", "pdfplumber", "fitz"
                library="fitz",
            ),
        ),
    )

    if embed == "openai":
        embed_cfg = lr.embedding_models.OpenAIEmbeddingsConfig()
    else:
        embed_cfg = lr.embedding_models.SentenceTransformerEmbeddingsConfig(
            model_type="sentence-transformer",
            model_name=embed,
        )

    match vecdb:
        case "lance" | "lancedb":
            config.vecdb = lr.vector_store.LanceDBConfig(
                collection_name="doc-chat-lancedb",
                replace_collection=True,
                storage_path=".lancedb/data/",
                embedding=embed_cfg,
            )
        case "qdrant" | "qdrantdb":
            config.vecdb = lr.vector_store.QdrantDBConfig(
                cloud=True,
                storage_path=".qdrant/doc-chat",
                embedding=embed_cfg,
            )
        case "chroma" | "chromadb":
            config.vecdb = lr.vector_store.ChromaDBConfig(
                storage_path=".chroma/doc-chat",
                embedding=embed_cfg,
            )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not nostream,
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
