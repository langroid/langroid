from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from langroid.agent.special.retriever_agent import (
    RecordDoc,
    RecordMetadata,
    RetrieverAgent,
    RetrieverAgentConfig,
)
from langroid.parsing.parser import ParsingConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER
from langroid.vector_store.qdrantdb import QdrantDBConfig


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


class _TestRetrieverAgentConfig(RetrieverAgentConfig):
    system_message: str = "You are a data scientist"
    user_message: str = """
        Your task is to match a profile description to a list of records in a table.
        """
    data: List[Dict[str, Any]]
    n_matches: int = 3  # num final matches to be picked by LLM
    vecdb: QdrantDBConfig = QdrantDBConfig(
        collection_name="test-retriever",
        storage_path=":memory:",
    )
    parsing: ParsingConfig = ParsingConfig(
        n_similar_docs=5,
    )
    cross_encoder_reranking_model = ""  # turn off cross-encoder reranking


class _TestRetrieverAgent(RetrieverAgent):
    def __init__(self, config: _TestRetrieverAgentConfig):
        super().__init__(config)
        self.config = config

    def get_records(self) -> Sequence[RecordDoc]:
        return [
            RecordDoc(
                content=", ".join(f"{k}={v}" for k, v in d.items()),
                metadata=RecordMetadata(id=i),
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
    response = agent.llm_response(query=query).content
    assert all([k in response for k in expected.split(",")])
    assert all([k not in response for k in not_expected.split(",")])
