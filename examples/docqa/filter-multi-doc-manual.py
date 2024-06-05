"""
Two-agent system to use to chat with multiple docs,
and use a combination of Filtering + RAG to answer questions,
where the filter is manually set via the LanceDocChatAgentConfig.filter field.

Works with LanceDB vector-db.

- Main agent takes user question, generates a QueryPlan consisting of
    - filter (SQL, to use with lanceDB)
    - possibly rephrased query

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

NOTES:
(1) The app works best with GPT4/Turbo, but results may be mixed with local LLMs.
You may have to tweak the system_message, use_message, and summarize_prompt
as indicated in comments below, to get good results.

"""

import typer
from rich import print
from rich.prompt import Prompt
import os

from pydantic import Field
import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.vector_store.lancedb import LanceDBConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MusicianMetadata(lr.DocMetaData):
    name: str = Field(..., description="The name of the musician.")
    birth_year: int = Field(..., description="The year the musician was born.")
    death_year: int = Field(..., description="The year the musician died.")
    type: str = Field(..., description="The type of musician, e.g. composer, musician.")
    genre: str = Field(..., description="The genre of the musician.")


class MusicianDocument(lr.Document):
    content: str
    metadata: MusicianMetadata


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
        chat_context_length=4096,  # adjust based on model
        timeout=90,
    )

    # Configs
    embed_cfg = OpenAIEmbeddingsConfig()

    # Get movies data
    COLLECTION = "chat-lance-music"
    ldb_dir = ".lancedb/data/musicians"
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name=COLLECTION,
        storage_path=ldb_dir,
        embedding=embed_cfg,
        replace_collection=False,
        document_class=MusicianDocument,
        flatten=False,
    )
    config = DocChatAgentConfig(
        name="MusicianBot",
        vecdb=ldb_cfg,
        n_query_rephrases=0,
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
        llm=llm_config,
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
            cache_type="fakeredis",
        )
    )

    print("[blue]Welcome to the Musician document-filtering chatbot!")

    # need a LanceDocChatAgent to use LanceRAgTaskCreator below
    agent = LanceDocChatAgent(config)

    # INGEST DOCS with META DATA
    beethoven_path = (
        "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven"  # or can be local dir
    )
    mozart_path = "https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart"
    bach_path = "https://en.wikipedia.org/wiki/Johann_Sebastian_Bach"
    hendrix_path = "https://en.wikipedia.org/wiki/Pink_Floyd"
    prince_path = "https://en.wikipedia.org/wiki/Prince_(musician)"
    jackson_path = "https://en.wikipedia.org/wiki/Michael_Jackson"

    paths = dict(
        beethoven=beethoven_path,
        mozart=mozart_path,
        bach=bach_path,
        hendrix=hendrix_path,
        prince=prince_path,
        jackson=jackson_path,
    )

    metadata = dict(
        beethoven=MusicianMetadata(
            name="Beethoven",
            birth_year=1770,
            death_year=1827,
            type="composer",
            genre="classical",
        ),
        mozart=MusicianMetadata(
            name="Mozart",
            birth_year=1756,
            death_year=1791,
            type="composer",
            genre="classical",
        ),
        bach=MusicianMetadata(
            name="Bach",
            birth_year=1685,
            death_year=1750,
            type="composer",
            genre="classical",
        ),
        hendrix=MusicianMetadata(
            name="Hendrix",
            birth_year=1942,
            death_year=1970,
            type="musician",
            genre="rock",
        ),
        prince=MusicianMetadata(
            name="Prince",
            birth_year=1958,
            death_year=2016,
            type="musician",
            genre="rock",
        ),
        jackson=MusicianMetadata(
            name="Jackson",
            birth_year=1958,
            death_year=2009,
            type="musician",
            genre="pop",
        ),
    )

    create_collection = True
    if COLLECTION in agent.vecdb.list_collections():
        replace = Prompt.ask(
            f"Collection {COLLECTION} already exists. Replace it? (y/n)",
            choices=["y", "n"],
            default="n",
        )
        if replace == "y":
            agent.vecdb.set_collection(COLLECTION, replace=True)
        else:
            create_collection = False
    if create_collection:
        print("[blue]Ingesting docs...")
        for musician in metadata:
            agent.ingest_doc_paths(
                [paths[musician]],  # all chunks of this doc will have same metadata
                metadata[musician],
            )
        print("[blue]Done ingesting docs")

    musician = Prompt.ask(
        "[blue]which musician would you like to ask about?",
        choices=list(metadata.keys()),
        default="beethoven",
    )
    print(f"[blue]You chose {metadata[musician].name}")
    # this filter setting will be used by the LanceDocChatAgent
    # to restrict the docs searched from the vector-db
    config.filter = f"metadata.name = '{metadata[musician].name}'"

    print("[blue]Reqdy for your questions...")
    task = lr.Task(
        agent,
        interactive=True,
    )
    task.run("Can you help me with some questions about musicians?")


if __name__ == "__main__":
    app()
