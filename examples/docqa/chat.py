"""
Single agent to use to chat with a Retrieval-augmented LLM.
Repeat: User asks question -> LLM answers.

Run like this, either with a document-path (can be URL, file-path, folder-path):

python3 examples/docqa/chat.py url-or-file-orfolder-path

(or run with no arguments to go through the dialog).

If a document-arg is provided, it will be ingested into the vector database.

To change the model, use the --model flag, e.g.:

python3 examples/docqa/chat.py --model ollama/mistral:7b-instruct-v0.2-q8_0

To change the embedding service provider, use the --embed and --embedconfig flags, e.g.:

For OpenAI
python3 examples/docqa/chat.py --embed openai

For Huggingface SentenceTransformers
python3 examples/docqa/chat.py --embed hf --embedconfig BAAI/bge-large-en-v1.5

For Llama.cpp Server
python3 examples/docqa/chat.py --embed llamacpp --embedconfig localhost:8000

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""

import os

import typer
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    doc: str = typer.Argument("", help="url, file-path or folder to chat about"),
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    vecdb: str = typer.Option(
        "qdrant", "--vecdb", "-v", help="vector db name (default: qdrant)"
    ),
    nostream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    embed_provider: str = typer.Option(
        "openai",
        "--embed",
        "-e",
        help="Embedding service provider",
        # openai, hf, llamacpp
    ),
    embed_config: str = typer.Option(
        None,
        "--embedconfig",
        "-ec",
        help="Embedding service host/sentence transformer model",
    ),
    # e.g. NeuML/pubmedbert-base-embeddings
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
        full_citations=True,
        hypothetical_answer=False,
        # how many sentences in each segment, for relevance-extraction:
        # increase this if you find that relevance extraction is losing context
        extraction_granularity=3,
        # for relevance extraction
        # relevance_extractor_config=None,  # set to None to disable relevance extraction
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=2,
        n_similar_chunks=5,
        n_relevant_chunks=5,
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
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # see here for possible values:
                # https://github.com/langroid/langroid/blob/main/langroid/parsing/parser.py
                library="pymupdf4llm",
            ),
        ),
    )

    match embed_provider:
        case "hf":
            embed_cfg = lr.embedding_models.SentenceTransformerEmbeddingsConfig(
                model_type="sentence-transformer",
                model_name=embed_config,
            )
        case "llamacpp":
            embed_cfg = lr.embedding_models.LlamaCppServerEmbeddingsConfig(
                api_base=embed_config,
                dims=768,  # Change this to match the dimensions of your embedding model
            )
        case "gemini":
            embed_cfg = lr.embedding_models.GeminiEmbeddingsConfig(
                model_type="gemini", dims=768
            )
        case _:
            embed_cfg = lr.embedding_models.OpenAIEmbeddingsConfig()

    match vecdb:
        case "lance" | "lancedb":
            config.vecdb = lr.vector_store.LanceDBConfig(
                collection_name="doc-chat-lancedb",
                storage_path=".lancedb/data/",
                embedding=embed_cfg,
            )
        case "qdrant" | "qdrantdb":
            config.vecdb = lr.vector_store.QdrantDBConfig(
                cloud=False,
                storage_path=".qdrant/doc-chat",
                embedding=embed_cfg,
            )
        case "chroma" | "chromadb":
            config.vecdb = lr.vector_store.ChromaDBConfig(
                storage_path=".chroma/doc-chat",
                embedding=embed_cfg,
            )
        case "weaviate" | "weaviatedb":
            config.vecdb = lr.vector_store.WeaviateDBConfig(
                embedding=embed_cfg,
            )
        case "pinecone" | "pineconedb":
            config.vecdb = lr.vector_store.PineconeDBConfig(
                collection_name="doc-chat-pinecone-serverless",
                embedding=embed_cfg,
            )
        case "postgres" | "postgresdb":
            config.vecdb = lr.vector_store.PostgresDBConfig(
                embedding=embed_cfg, cloud=True
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

    if doc:
        # TODO - could save time by checking whether we've already ingested this doc(s)
        agent.ingest_doc_paths([doc])
    else:
        agent.user_docs_ingest_dialog()

    print("[cyan]Enter x or q to quit")

    task = lr.Task(
        agent,
        system_message="You are a helpful assistant, "
        "answering questions about some docs",
    )
    task.run()


if __name__ == "__main__":
    app()
