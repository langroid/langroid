from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.retriever_agent import RetrieverAgent
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import ParsingConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig


def gen_data(size: int) -> List[Dict[str, Any]]:
    # Create a list of states
    states = ["CA", "TX"]

    # Generate random age between 18 and 100
    ages = np.random.choice([18, 80], size)

    # Generate random gender
    genders = np.random.choice(["male", "female"], size)

    # Generate random state
    states_col = np.random.choice(states, size)

    # Generate random income between 30000 and 150000
    incomes = np.random.choice([15_000, 100_000], size)

    data = [
        dict(
            age=ages[i],
            gender=genders[i],
            state=states_col[i],
            income=incomes[i],
        )
        for i in range(size)
    ]

    # add special record to test if it can be found
    data.append(
        dict(
            age=100,
            gender="male",
            state="NJ",
            income=1_000_000,
        )
    )

    return data


class _TestRetrieverAgentConfig(DocChatAgentConfig):
    system_message: str = "You are a data scientist"
    user_message: str = """
        Your task is to match a profile description to a list of records in a table.
        """
    data: Optional[List[Dict[str, Any]]] = None
    retrieve_only: bool = True
    retrieval_granularity: int = -1  # extract whole content
    n_similar_chunks: int = 5
    n_relevant_chunks: int = 5
    vecdb: QdrantDBConfig = QdrantDBConfig(
        collection_name="test-retriever",
        storage_path=":memory:",
    )
    parsing: ParsingConfig = ParsingConfig()
    cross_encoder_reranking_model: str = ""  # turn off cross-encoder reranking


class _TestRetrieverAgent(RetrieverAgent):
    def __init__(self, config: _TestRetrieverAgentConfig):
        super().__init__(config)
        self.config = config

    def get_records(self) -> Sequence[Document]:
        return [
            Document(
                content=", ".join(f"{k}={v}" for k, v in d.items()),
                metadata=DocMetaData(id=str(i)),
            )
            for i, d in enumerate(self.config.data)
        ]


dicts = gen_data(100)
cfg = _TestRetrieverAgentConfig(
    data=dicts,
)
agent = _TestRetrieverAgent(cfg)
agent.ingest()


@pytest.mark.parametrize(
    "query,expected,not_expected",
    [
        (
            "Men in CA who are over 75",
            "age=80,gender=male,state=CA",
            "age=18,gender=female,state=TX",
        ),
        (
            "People earning at least 100k",
            "income=100000",
            "income=15000",
        ),
        (
            "People earning over 100k in CA",
            "income=100000,state=CA",
            "state=TX",
        ),
        (
            "Folks living in CA",
            "state=CA",
            "state=TX,state=NJ",
        ),
        (
            "Canada residents",
            NO_ANSWER,
            "age,gender,state,income",
        ),
        (
            "People living in New Jersey",
            "age=100,gender=male,state=NJ",
            "state=CA,state=TX",
        ),
    ],
)
def test_retriever_agent(
    test_settings: Settings,
    query: str,
    expected: str,
    not_expected: str,
) -> None:
    set_global(test_settings)
    response = agent.llm_response(message=query).content
    assert all([k in response for k in expected.split(",")])
    assert all([k not in response for k in not_expected.split(",")])


embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)


class MyDocMetaData(DocMetaData):
    id: str


class MyDoc(Document):
    content: str
    metadata: MyDocMetaData


@pytest.fixture(scope="function")
def vecdb(request) -> VectorStore:
    if request.param == "qdrant_local":
        qd_dir = ":memory:"
        qd_cfg = QdrantDBConfig(
            cloud=False,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=qd_dir,
            embedding=embed_cfg,
        )
        qd = QdrantDB(qd_cfg)
        yield qd
        return

    if request.param == "chroma":
        cd_dir = ".chroma/" + embed_cfg.model_type
        rmdir(cd_dir)
        cd_cfg = ChromaDBConfig(
            collection_name="test-" + embed_cfg.model_type,
            storage_path=cd_dir,
            embedding=embed_cfg,
        )
        cd = ChromaDB(cd_cfg)
        yield cd
        rmdir(cd_dir)
        return

    if request.param == "lancedb":
        ldb_dir = ".lancedb/data/" + embed_cfg.model_type
        rmdir(ldb_dir)
        ldb_cfg = LanceDBConfig(
            cloud=False,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=ldb_dir,
            embedding=embed_cfg,
            document_class=MyDoc,  # IMPORTANT, to ensure table has full schema!
        )
        ldb = LanceDB(ldb_cfg)
        yield ldb
        rmdir(ldb_dir)
        return


summaries = SimpleNamespace(
    ENTROPY="A story exploring the concept of entropy and the end of the universe.",
    HARRY_POTTER="The adventures of a young wizard at a magical school.",
    BIG_BROTHER="A dystopian novel about a totalitarian regime and what freedom means.",
    LOTR="An epic fantasy tale of a quest to destroy a powerful ring.",
    TIME_MACHINE="A science fiction novel about time travel and its consequences.",
)

data = {
    "id": ["A100", "B200", "C300", "D400", "E500"],
    "year": [1955, 1977, 1989, 2001, 2015],
    "summary": list(vars(summaries).values()),
}

df = pd.DataFrame(data)


@pytest.mark.parametrize("metadata", [[], ["id", "year"], ["year"]])
@pytest.mark.parametrize("vecdb", ["lancedb", "qdrant_local", "chroma"], indirect=True)
def test_retriever_agent_from_df(
    test_settings: Settings,
    vecdb,
    metadata,
):
    """Check we can ingest from a dataframe and run queries."""
    set_global(test_settings)

    agent_cfg = _TestRetrieverAgentConfig()
    agent = RetrieverAgent(agent_cfg)
    agent.vecdb = vecdb
    agent.clear()
    agent.ingest_dataframe(df, content="summary", metadata=metadata)
    response = agent.llm_response(
        """
        A movie about the end of the universe or about a magical school.
        """
    )
    # Check that the ENTIRE description is returned
    assert summaries.ENTROPY in response.content
    assert summaries.HARRY_POTTER in response.content
