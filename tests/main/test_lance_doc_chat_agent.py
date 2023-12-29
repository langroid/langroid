import gzip
import json

import pandas as pd
import pytest
from pydantic import Field

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import (
    LanceDocChatAgent,
    LanceRAGTaskCreator,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.repo_loader import RepoLoader
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
            "The Godfeather is a 1972 American crime " "film directed by Frank Copula.",
            "The Lamb Shank Redemption is a 1994 American drama "
            "film directed by Garth Brook.",
        ],
        "year": [1999, 1968, 1972, 1994],
        "director": [
            "Jomes Winkowski",
            "Stanley Hendrick",
            "Frank Copula",
            "Garth Brook",
        ],
        "genre": ["Science Fiction", "Science Fiction", "Crime", "Drama"],
        "rating": [8.7, 8.9, 9.2, 9.3],
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
@pytest.mark.parametrize("flatten", [True, False])
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
        storage_path=ldb_dir,
        embedding=embed_cfg,
        document_class=FlatMovieDoc,
        flatten=flatten,
    )

    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
    )
    agent = LanceDocChatAgent(cfg)

    # convert df to list of dicts
    doc_dicts = df.to_dict(orient="records")
    # convert doc_dicts to list of FlatMovieDocs
    docs = [FlatMovieDoc(**d) for d in doc_dicts]
    agent.ingest_docs(docs, split=False)

    task = LanceRAGTaskCreator.new(agent, interactive=False)

    result = task.run(query)
    assert expected in result.content


def parse_gz(path):
    g = gzip.open(path, "rb")
    for x in g:
        yield json.loads(x)


def getDF_gz(path):
    i = 0
    df = {}
    for d in parse_gz(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def parse(path):
    with open(path, "r") as file:
        for line in file:
            try:
                dct = json.loads(line)
                if dct.get("style") is None:
                    dct["style"] = {}
                if dct.get("image") is None:
                    dct["image"] = []
                if isinstance(dct.get("vote", "0"), str):
                    dct["vote"] = int(dct.get("vote", "0").replace(",", ""))
                yield dct
            except json.JSONDecodeError:
                pass


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


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
    )
    agent = LanceDocChatAgent(cfg)

    # load github issues from a repo
    repo_loader = RepoLoader("jmorganca/ollama")
    issues = repo_loader.get_issues(k=100)
    issue_dicts = [iss.dict() for iss in issues]
    df = pd.DataFrame(issue_dicts)
    metadata_cols = []
    agent.ingest_dataframe(df, content="text", metadata=metadata_cols)
    task = LanceRAGTaskCreator.new(agent, interactive=False)
    result = task.run(
        """
        Tell me about some open issues related to JSON
        """
    )
    # check there is non-empty response content
    assert result is not None and len(result.content) > 10
