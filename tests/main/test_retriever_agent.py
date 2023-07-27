import os
import warnings
from typing import List

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
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

storage_path = ".qdrant/testdata1"
rmdir(storage_path)


class PeopleMetadata(RecordMetadata):
    source: str = None
    id: int


class PeoplyRecord(RecordDoc):
    metadata: PeopleMetadata


class _TesrRetrieverAgentConfig(RetrieverAgentConfig):
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="test-data",
        storage_path=storage_path,
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    )

    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_similar_docs=2,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


set_global(
    Settings(
        cache=True,
        gpt3_5=True,
    )
)  # allow cacheing

people = [
    {"id": 1, "age": 35, "gender": "male", "height": "6.1ft"},
    {"id": 2, "age": 33, "gender": "female", "height": "5.7ft"},
    {"id": 3, "age": 42, "gender": "female", "height": "5.8ft"},
    {"id": 4, "age": 23, "gender": "male", "height": "5.9ft"},
    {"id": 5, "age": 31, "gender": "female", "height": "5.6ft"},
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class _TestRetrieverAgent(RetrieverAgent):
    def __init__(self, config: _TesrRetrieverAgentConfig):
        super().__init__(config)
        self.config = config

    def get_records(self) -> List[PeoplyRecord]:
        people_rec = []
        row_num = 0
        for person in people:
            full_row = f"{person['age']}|{person['gender']}|{person['height']}"
            row_num += 1
            meta = PeopleMetadata(id=int(person["id"]), source=f"people_{row_num}")
            people_rec.append(PeoplyRecord(content=full_row, metadata=meta))
        return people_rec


config = _TesrRetrieverAgentConfig()
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
        "Can you give me list of males over 20, under 5.5 feet tall",
        "No satisfying document",
    ),
    (
        "Can you give me list of males over 30, more 6 feet tall",
        "6.1ft",
    ),
    (
        (
            "Can you give me list of females over 30 but less than 40, more 5.5"
            " feet tall"
        ),
        "5.7ft,5.6ft",
    ),
]


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_retriever_chat_agent(test_settings: Settings, query: str, expected: str):
    set_global(test_settings)
    ans = agent.llm_response(query).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


def test_get_nearest_docs():
    query = (
        "Can you give me list of males whose age less 30 and their height is"
        " MORE 5.5ft"
    )
    ans = agent.get_nearest_docs(query)
    assert len(ans) == 2


def test_get_relevant_docs():
    query = (
        "Can you give me list of males whose age less 30 and their height is"
        " MORE 5.5ft"
    )
    ans = agent.get_relevant_docs(query)
    assert len(ans) == 1


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
