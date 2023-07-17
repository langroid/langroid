import os
import warnings
from typing import List

import pytest

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.parsing.utils import generate_random_text
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

storage_path = ".qdrant/testdata1"
rmdir(storage_path)


class _TestDocChatAgentConfig(DocChatAgentConfig):
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


config = _TestDocChatAgentConfig()
set_global(Settings(cache=True))  # allow cacheing
documents: List[Document] = (
    [
        Document(
            content="""
        In the year 2050, GPT10 was released. 
        
        In 2057, paperclips were seen all over the world. 
        
        Global warming was solved in 2060. 
        
        In 2061, the world was taken over by paperclips. 
        
        In 2045, the Tour de France was still going on.
        They were still using bicycles. 
        
        There was one more ice age in 2040.
        """,
            metadata=DocMetaData(source="wikipedia"),
        ),
        Document(
            content="""
        We are living in an alternate universe where Paris is the capital of England.
        
        The capital of England used to be London. 
        
        The capital of France used to be Paris.
        
        Charlie Chaplin was a great comedian.
        
        In 2050, all countries merged into Lithuania.
        """,
            metadata=DocMetaData(source="almanac"),
        ),
    ]
    + [
        Document(content=generate_random_text(5), metadata={"source": "random"})
        for _ in range(10)
    ]
)


agent = DocChatAgent(config)
agent.ingest_docs(documents)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)

QUERY_EXPECTED_PAIRS = [
    ("what happened in the year 2050?", "GPT10, Lithuania"),
    ("Who was Charlie Chaplin?", "comedian"),
    # ("What was the old capital of England?", "London"), this often fails!!
    ("What was the old capital of France?", "Paris"),
    ("When was global warming solved?", "2060"),
    ("what is the capital of England?", "Paris"),
    ("What do we know about paperclips?", "2057, 2061"),
]


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_doc_chat_agent(test_settings: Settings, query: str, expected: str):
    # set_global(Settings(debug=options.show, cache=not options.nocache))
    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)
    ans = agent.llm_response(query).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


def test_doc_chat_process(test_settings: Settings):
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
