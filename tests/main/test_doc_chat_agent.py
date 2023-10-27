import os
import warnings
from types import SimpleNamespace
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
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

storage_path = ":memory:"


class _TestDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        collection_name="test-data",
        replace_collection=True,
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
        n_similar_docs=3,
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
        
        Charlie Chaplin was born in 1889.
        
        Beethoven was born in 1770.
        
        In the year 2050, all countries merged into Lithuania.
        """,
            metadata=DocMetaData(source="almanac"),
        ),
    ]
    + [Document(content=generate_random_text(5), metadata={"source": "random"})] * 100
)


@pytest.fixture
def agent():
    agent = DocChatAgent(config)
    agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return agent


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)

QUERY_EXPECTED_PAIRS = [
    ("what happened in the year 2050?", "GPT10, Lithuania"),
    ("what is the capital of England?", "Paris"),
    ("Who was Charlie Chaplin?", "comedian"),
    ("What used to be capital of France?", "Paris"),
    ("When was global warming solved?", "2060"),
    ("What do we know about paperclips?", "2057, 2061"),
]


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_doc_chat_agent_llm(test_settings: Settings, agent, query: str, expected: str):
    """
    Test directly using `llm_response` method of DocChatAgent.
    """

    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)
    agent.config.conversation_mode = False
    ans = agent.llm_response(query).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


def test_doc_chat_agent_task(test_settings: Settings, agent):
    """
    Test DocChatAgent wrapped in a Task.
    """
    set_global(test_settings)
    agent.config.conversation_mode = True
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


@pytest.mark.parametrize("conv_mode", [True, False])
def test_doc_chat_followup(test_settings: Settings, agent, conv_mode: bool):
    """
    Test whether follow-up question is handled correctly.
    """
    agent.config.conversation_mode = conv_mode
    set_global(test_settings)
    task = Task(
        agent,
        default_human_response="",
        restart=True,
        single_round=True,
    )
    result = task.run("Who was Charlie Chaplin?")
    assert "comedian" in result.content.lower()

    result = task.run("When was he born?")
    assert "1889" in result.content


# setup config for retrieval test, with n_neighbor_chunks=2
# and parser.n_neighbor_ids = 5
class _MyDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    n_neighbor_chunks = 2
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        collection_name="test-data",
        replace_collection=True,
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
        n_neighbor_ids=5,
    )


@pytest.mark.parametrize("conv_mode", [True, False])
def test_doc_chat_retrieval(test_settings: Settings, agent, conv_mode: bool):
    """
    Test retrieval of relevant doc-chunks
    """
    agent = DocChatAgent(_MyDocChatAgentConfig())
    agent.config.conversation_mode = conv_mode

    set_global(test_settings)

    phrases = SimpleNamespace(
        CATS="Cats are quiet and clean.",
        DOGS="Dogs are loud and messy.",
        PIGS="Pigs cannot fly.",
        GIRAFFES="Giraffes are tall and vegetarian.",
        BATS="Bats are blind.",
        COWS="Cows are peaceful.",
        GIRAFFES2="Giraffes are really strange animals.",
        HYENAS="Hyenas are dangerous and fast.",
        ZEBRAS="Zebras are bizarre with stripes.",
    )
    text = "\n\n".join(vars(phrases).values())
    agent.ingest_docs([Document(content=text, metadata={"source": "animals"})])
    results = agent.get_relevant_chunks("What are giraffes like?")

    all_but_cats = [p for p in vars(phrases).values() if "Cats" not in p]
    # check that each phrases occurs in exactly one result
    assert (
        sum(p in r.content for p in all_but_cats for r in results)
        == len(vars(phrases)) - 1
    )
