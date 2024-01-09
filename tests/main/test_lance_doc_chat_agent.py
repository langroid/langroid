import pandas as pd
import pytest
from pydantic import Field

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.agent.special.lance_rag.lance_rag_task import LanceRAGTaskCreator
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER
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


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "Which Science Fiction movie was directed by Winkowski?",
            "The Vector",
        ),
        (
            "Which Crime movie had a rating over 9?",
            "Godfeather",
        ),
        (
            "What was the Science Fiction movie directed by Stanley Hendrick?",
            "Sparse Odyssey",
        ),
    ],
)
@pytest.mark.parametrize("flatten", [True, False])
def test_lance_doc_chat_agent(
    test_settings: Settings,
    flatten: bool,
    query: str,
    expected: str,
):
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
    task = LanceRAGTaskCreator.new(agent, interactive=False)

    result = task.run(query)
    assert NO_ANSWER in result.content or expected in result.content


# dummy pandas dataframe from text
df = pd.DataFrame(
    {
        "content": [
            "The Vector is a 1999 science fiction action film written "
            "and directed by Jomes Winkowski.",
            "Sparse Odyssey is a 1968 science fiction film produced "
            "and directed by Stanley Hendrick.",
            "The Godfeather is a 1972 American crime film directed by Frank Copula.",
            "The Lamb Shank Redemption is a 1994 American drama "
            "film directed by Garth Brook about a prison escape.",
            "Escape from Alcoona is a 1979 American prison action film  "
            "directed by Dan Seagull.",
        ],
        "year": [1999, 1968, 1972, 1994, 1979],
        "director": [
            "Jomes Winkowski",
            "Stanley Hendrick",
            "Frank Copula",
            "Garth Brook",
            "Dan Seagull",
        ],
        "genre": ["Science Fiction", "Science Fiction", "Crime", "Crime", "Crime"],
        "rating": [8, 10, 9.2, 8.7, 9.0],
    }
)


class FlatMovieDoc(Document):
    content: str = Field(..., description="A short description of the movie.")
    year: int = Field(..., description="The year the movie was released.")
    director: str = Field(
        ..., description="The Full Name of the director of the movie."
    )
    genre: str = Field(..., description="The genre of the movie.")
    rating: float = Field(..., description="The rating of the movie.")
    metadata: DocMetaData = DocMetaData()


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "Which movie about about incarceration or jails is rated highest?",
            "Alcoona",
        ),
        (
            "Which Science Fiction movie is rated highest?",
            "Odyssey",
        ),
        (
            "Average rating of Science Fiction movies?",
            "9",
        ),
        (
            "Tell me about a Crime movie rated over 9",
            "Godfeather",
        ),
        (
            "What was the Science Fiction movie directed by Stanley Hendrick?",
            "Sparse Odyssey",
        ),
        (
            "Which Science Fiction movie was directed by Winkowski?",
            "The Vector",
        ),
    ],
)
@pytest.mark.parametrize("flatten", [False, True])
def test_lance_doc_chat_agent_df(
    test_settings: Settings,
    flatten: bool,
    query: str,
    expected: str,
):
    set_global(test_settings)

    ldb_dir = ".lancedb/data/test-2"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="test-lance-2",
        replace_collection=True,
        storage_path=ldb_dir,
        embedding=embed_cfg,
        document_class=FlatMovieDoc,
        flatten=flatten,
    )

    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
        add_fields_to_content=["year", "director", "genre"],
        filter_fields=["year", "director", "genre", "rating"],
    )
    agent = LanceDocChatAgent(cfg)

    # convert df to list of dicts
    doc_dicts = df.to_dict(orient="records")
    # convert doc_dicts to list of FlatMovieDocs
    docs = [FlatMovieDoc(**d) for d in doc_dicts]
    agent.ingest_docs(docs, split=False)

    task = LanceRAGTaskCreator.new(agent, interactive=False)

    result = task.run(query)
    assert NO_ANSWER in result.content or expected in result.content


def test_lance_doc_chat_df_direct(test_settings: Settings):
    set_global(test_settings)

    ldb_dir = ".lancedb/data/gh-issues"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="test-lance-gh-issues",
        storage_path=ldb_dir,
        embedding=embed_cfg,
    )

    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
        add_fields_to_content=["state", "year"],
        filter_fields=["state", "year"],
    )
    agent = LanceDocChatAgent(cfg)

    df = pd.read_csv("tests/main/data/github-issues.csv")
    # only get year, state, text columns
    df = df[["year", "state", "text"]]
    agent.ingest_dataframe(df, content="text", metadata=[])
    task = LanceRAGTaskCreator.new(agent, interactive=False)
    result = task.run(
        """
        Tell me about some open issues from year 2023 related to JSON
        """
    )
    # check there is non-empty response content
    assert result is not None and len(result.content) > 10
