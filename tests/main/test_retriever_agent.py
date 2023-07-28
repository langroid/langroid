import random
import warnings
from typing import Dict, List, Union

import pytest

from langroid.agent.special.retriever_agent import (
    RecordDoc,
    RecordMetadata,
    RetrieverAgent,
    RetrieverAgentConfig,
)
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.system import rmdir
from langroid.vector_store.qdrantdb import QdrantDBConfig

storage_path = ".qdrant/testdata1"
rmdir(storage_path)


def generate_data(size: int) -> List[Dict[str, Union[int, float, str]]]:
    # Create a list of states
    states = ["CA", "TX"]

    # Generate random age between 18 and 100
    ages = [random.randint(18, 100) for _ in range(size)]

    # Generate random gender
    genders = [random.choice(["Male", "Female"]) for _ in range(size)]

    # Generate random state
    states_col = [random.choice(states) for _ in range(size)]

    # Generate random height between 4.0 and 6.5
    heights = [round(random.uniform(4.0, 6.5), 2) for _ in range(size)]

    random_people = [
        {
            "id": i + 1,
            "age": ages[i],
            "gender": genders[i],
            "state": states_col[i],
            "height": heights[i],
        }
        for i in range(size)
    ]

    return random_people


class PeopleMetadata(RecordMetadata):
    source: str = None
    id: int


class PeoplyRecord(RecordDoc):
    metadata: PeopleMetadata


set_global(
    Settings(
        cache=True,
        gpt3_5=True,
    )
)  # allow cacheing

known_people = [
    {"id": 102, "age": 99, "gender": "male", "state": "NY", "height": "6.1ft"},
    {
        "id": 103,
        "age": 98,
        "gender": "female",
        "state": "NY",
        "height": "5.7ft",
    },
    {
        "id": 104,
        "age": 97,
        "gender": "female",
        "state": "NY",
        "height": "5.8ft",
    },
]

people = generate_data(100) + known_people


class _TestRetrieverAgent(RetrieverAgent):
    def __init__(self, config: RetrieverAgentConfig):
        super().__init__(config)
        self.config = config

    def get_records(self) -> List[PeoplyRecord]:
        people_rec = []
        row_num = 0
        for person in people:
            full_row = f"""
            {person['age']}|{person['gender']}|{person['state']}|{person['height']}
            """
            row_num += 1
            meta = PeopleMetadata(id=int(person["id"]), source=f"people_{row_num}")
            people_rec.append(PeoplyRecord(content=full_row, metadata=meta))
        return people_rec


# Now create an instance of RetrieverAgentConfig
config = RetrieverAgentConfig(
    debug=False,
    conversation_mode=True,
    stream=True,
    vecdb=QdrantDBConfig(
        type="qdrant",
        collection_name="test-data",
        storage_path=storage_path,
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    ),
    llm=OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    ),
    parsing=ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_similar_docs=2,
    ),
    prompts=PromptsConfig(
        max_tokens=1000,
    ),
)

agent = _TestRetrieverAgent(config)

agent.ingest()


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)

QUERY_EXPECTED_PAIRS = [
    (
        "males from NY",
        "99|male|NY|6.1ft",
    ),
    (
        "people from NY and shorter than 6ft",
        "5.8ft",
    ),
    (
        "females from NY",
        "5.8ft,5.7ft",
    ),
]


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_retriever_chat_agent(test_settings: Settings, query: str, expected: str):
    set_global(test_settings)
    ans = agent.llm_response(query).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


def test_get_nearest_docs():
    query = "people from NY"
    ans = agent.get_nearest_docs(query)
    id = [103, 104]
    assert all(rec.metadata.id in id for rec in ans)


def test_get_relevant_docs():
    query = "males whose age less 30 and their height is MORE 6ft"
    ans = agent.get_relevant_docs(query)
    assert "6.1ft" in ans[0].content


def test_retreiver_chat_process(test_settings: Settings):
    set_global(test_settings)
    task = Task(agent, restart=True)
    task.init()
    # LLM responds to Sys msg, initiates conv, says thank you, etc.
    task.step()
    for q, expected in QUERY_EXPECTED_PAIRS:
        agent.default_human_response = q
        task.step()  # user asks `q`
        task.step()  # LLM answers
        ans = task.pending_message.content
        expected = [e.strip() for e in expected.split(",")]
        assert all([e in ans for e in expected])
        assert task.pending_message.metadata.sender == Entity.LLM
