"""
Single agent to chat with multiple docs, with filtering based on user query.

- user asks a query containing an implicit filter,
  e.g. "what is the birth year of Beethoven?", implying a filter on
  docs where metadata.name == "Beethoven".
- DocChatAgent answers question using RAG restricted to the filtered docs.

"""

import json
import os
from typing import Optional

from fire import Fire
from rich import print
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from pydantic import Field
from langroid.utils.configuration import Settings, set_global
from langroid.utils.pydantic_utils import temp_update
from langroid.vector_store.lancedb import LanceDBConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECDB = "qdrant"  # or "lance"


class MusicianMetadata(lr.DocMetaData):
    name: str = Field(..., description="The name of the musician.")
    birth_year: int = Field(..., description="The year the musician was born.")
    death_year: int = Field(..., description="The year the musician died.")
    type: str = Field(..., description="The type of musician, e.g. composer, musician.")
    genre: str = Field(..., description="The genre of the musician.")


class MusicianDocument(lr.Document):
    content: str
    metadata: MusicianMetadata


class QueryPlanTool(lr.ToolMessage):
    request: str = "query_plan"
    purpose: str = """
        Given a user's query, generate a query plan consisting of the <name>
        the user is asking about, (which will be used to filter the document-set)
        and a possibly modified <query> (e.g. it may not need to contain the <name>).
        """
    name: str
    query: str


class FilterDocAgent(lr.agent.special.DocChatAgent):
    def llm_response(
        self,
        message: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        """Override DocChatAgent's default method,
        to call ChatAgent's llm_response, so it emits the QueryPlanTool"""
        return lr.ChatAgent.llm_response(self, message)

    def query_plan(self, msg: QueryPlanTool) -> str:
        """Handle query plan tool"""
        # Note the filter syntax depends on the type of underlying vector-db
        if VECDB == "lance":
            name_filter = f"metadata.name=='{msg.name}'"  # SQL-like syntax
        else:
            # for qdrant use this:
            name_filter_dict = dict(
                should=[dict(key="metadata.name", match=dict(value=msg.name))]
            )
            name_filter = json.dumps(name_filter_dict)
        with temp_update(self.config, {"filter": name_filter}):
            # restrict the document-set used for keyword and other non-vector
            # similarity
            self.setup_documents(filter=name_filter)
            extracts = self.get_relevant_chunks(msg.query)
        prompt = f"""
        Answer the QUESTION below based on the following EXTRACTS:
        
        EXTRACTS:
        {extracts}
        
        QUESTION: {msg.query}
        """
        response = lr.ChatAgent.llm_response(self, prompt)
        return response.content


def main(
    debug: bool = False,
    nocache: bool = False,
    model: str = "",
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=16_000,  # adjust based on model
        timeout=90,
    )

    # Configs
    embed_cfg = OpenAIEmbeddingsConfig()

    # Get movies data
    COLLECTION = "chat-filter-doc"
    # Note the filter syntax depends on the type of vecdb
    if VECDB == "lance":
        vecdb_cfg = LanceDBConfig(
            cloud=False,
            collection_name=COLLECTION,
            storage_path=".lance/data",
            embedding=embed_cfg,
            replace_collection=False,
            document_class=MusicianDocument,
        )
    else:
        vecdb_cfg = QdrantDBConfig(
            embedding=embed_cfg,
            cloud=False,
            storage_path=":memory:",  # in-memory storage
            collection_name=COLLECTION,
        )
    config = DocChatAgentConfig(
        name="MusicianBot",
        system_message="""
        You will respond to a query in 2 ways:
        
        - if you receive just a QUERY about a musician, 
            you must use the `query_plan` tool/function to generate a query plan.
        - if you receive document EXTRACTS followed by a QUESTION,
            simply answer the question based on the extracts.
            
        Start by asking the user what help they need.
        """,
        vecdb=vecdb_cfg,
        n_query_rephrases=0,
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
        n_similar_chunks=3,
        n_relevant_chunks=3,
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
            cache_type="fakeredis",
        )
    )

    print("[blue]Welcome to the Musician document-filtering chatbot!")

    agent = FilterDocAgent(config)
    agent.enable_message(QueryPlanTool)

    # INGEST DOCS with META DATA
    beethoven_path = (
        "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven"  # or can be local dir
    )

    bach_path = "https://en.wikipedia.org/wiki/Johann_Sebastian_Bach"

    paths = dict(
        beethoven=beethoven_path,
        bach=bach_path,
    )

    metadata = dict(
        beethoven=MusicianMetadata(
            name="Beethoven",
            birth_year=1770,
            death_year=1827,
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

    print("[blue]Reqdy for your questions...")
    task = lr.Task(agent, interactive=True)
    task.run()


if __name__ == "__main__":
    Fire(main)
