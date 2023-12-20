import pytest
from pydantic import Field

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import (
    LanceDocChatAgent,
    LanceRAGTaskCreator,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.configuration import Settings, set_global
from langroid.utils.system import rmdir
from langroid.vector_store.lancedb import LanceDBConfig


class MovieMetadata(DocMetaData):
    # Field(..., ) are optional but can help the LLM
    year: int = Field(..., description="The year the movie was released.")
    director: str = Field(
        ..., description="The Full Name of the director of the movie."
    )
    genre: str = Field(..., description="The genre of the movie.")
    rating: float = Field(..., description="The rating of the movie.")
    metadata: DocMetaData = DocMetaData()


class MovieDoc(Document):
    content: str = Field(..., description="A short description of the movie.")
    metadata: MovieMetadata


movie_docs = [
    MovieDoc(
        content="""
        The Vector is a 1999 science fiction action film written and 
        directed by Jomes Winkowski.
        """,
        metadata=MovieMetadata(
            year=1999,
            director="Jomes Winkowski",
            genre="Science Fiction",
            rating=8.7,
        ),
    ),
    MovieDoc(
        content="""
        Sparse Odyssey is a 1968 science fiction film produced and directed
        by Stanley Hendrick.
        """,
        metadata=MovieMetadata(
            year=1968, director="Stanley Hendrick", genre="Science Fiction", rating=8.9
        ),
    ),
    MovieDoc(
        content="""
        The Godfeather is a 1972 American crime film directed by Frank Copula.
        """,
        metadata=MovieMetadata(
            year=1972, director="Frank Copula", genre="Crime", rating=9.2
        ),
    ),
    MovieDoc(
        content="""
        The Lamb Shank Redemption is a 1994 American drama film directed by Garth Brook.
        """,
        metadata=MovieMetadata(
            year=1994, director="Garth Brook", genre="Drama", rating=9.3
        ),
    ),
]

embed_cfg = OpenAIEmbeddingsConfig()


@pytest.mark.parametrize("flatten", [True, False])
def test_lance_doc_chat_agent(test_settings: Settings, flatten: bool):
    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)

    ldb_dir = ".lancedb/data/test-2"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="test-lance-2",
        storage_path=ldb_dir,
        embedding=embed_cfg,
        document_class=MovieDoc,
        flatten=flatten,
    )

    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
    )
    agent = LanceDocChatAgent(cfg)
    agent.ingest_docs(movie_docs, split=False)
    task = LanceRAGTaskCreator.new(agent)

    # question on filtered docs
    query = "Which Crime movie had a rating over 9?"
    result = task.run(query)
    assert "Godfeather" in result.content

    # question on filtered docs
    query = "What was the Science Fiction movie directed by Stanley Hendrick?"
    result = task.run(query)
    assert "Sparse Odyssey" in result.content

    # ask with only director first name, then initial filter may be wrong
    # but the LLM should re-try with an approximate match.
    query = "Which Science Fiction movie was directed by Winkowski?"
    result = task.run(query)
    assert "The Vector" in result.content
