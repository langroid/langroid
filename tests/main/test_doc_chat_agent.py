import os
import warnings
from types import SimpleNamespace
from typing import List, Optional

import pandas as pd
import pytest

from langroid import ChatDocument
from langroid.agent.batch import run_batch_task_gen, run_batch_tasks
from langroid.agent.chat_agent import ChatAgent
from langroid.agent.special.doc_chat_agent import (
    CHUNK_ENRICHMENT_DELIMITER,
    ChunkEnrichmentAgentConfig,
    DocChatAgent,
    DocChatAgentConfig,
    RetrievalTool,
    _append_metadata_source,
)
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.parsing.utils import generate_random_text
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.utils.output.citations import extract_markdown_references
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore, VectorStoreConfig
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)


class MyDocMetaData(DocMetaData):
    id: str


class MyDoc(Document):
    content: str
    metadata: MyDocMetaData


documents: List[Document] = [
    Document(
        content="""
        In the year 2050, GPT10 was released. 
        
        In 2057, paperclips were seen all over the world. 
        
        Global warming was solved in 2060. 
        
        In 2045, the Tour de France was still going on.
        They were still using bicycles. 
        
        There was one more ice age in 2040.
        """,
        metadata=DocMetaData(source="wikipedia"),
    ),
    Document(
        content="""
    Winegarten is the capital of a new country called NeoGlobal.
        
    Charlie Foster was a great comedian.
        
    Charlie Foster was born in 1889.
        
    Beethoven was born in 1770.
        
    In the year 2050, all countries merged into Lithuania.
    """,
        metadata=DocMetaData(source="almanac"),
    ),
]

QUERY_EXPECTED_PAIRS = [
    ("what is the capital of NeoGlobal?", "Winegarten"),
    ("Who was Charlie Foster?", "comedian"),
    ("When was global warming solved?", "2060"),
    ("What do we know about paperclips?", "2057"),
]

for _ in range(100):
    documents.append(
        Document(
            content=generate_random_text(5),
            metadata=DocMetaData(source="random"),
        )
    )


@pytest.fixture(scope="function")
def vecdb(test_settings: Settings, request) -> VectorStore:
    set_global(test_settings)
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

    if request.param == "qdrant_cloud":
        qd_dir = ".qdrant/cloud/test-" + embed_cfg.model_type
        qd_cfg_cloud = QdrantDBConfig(
            cloud=True,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=qd_dir,
            embedding=embed_cfg,
        )
        qd_cloud = QdrantDB(qd_cfg_cloud)
        yield qd_cloud
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


class _TestDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig | None = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )

    n_similar_chunks = 3
    n_relevant_chunks = 3
    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.SIMPLE,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


config = _TestDocChatAgentConfig()
set_global(Settings(cache=True))  # allow cacheing


@pytest.fixture(scope="function")
def agent(test_settings: Settings, vecdb) -> DocChatAgent:
    set_global(test_settings)
    agent = DocChatAgent(config)
    agent.vecdb = vecdb
    agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return agent


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)


@pytest.mark.parametrize(
    "vecdb", ["lancedb", "qdrant_local", "qdrant_cloud", "chroma"], indirect=True
)
@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_doc_chat_agent_llm(test_settings: Settings, agent, query: str, expected: str):
    """
    Test directly using `llm_response` method of DocChatAgent.
    """

    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)
    agent.config.conversation_mode = False
    result = agent.llm_response(query)
    ans = result.content
    refs = extract_markdown_references(ans)
    sources = extract_markdown_references(result.metadata.source)
    assert refs == sources
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


@pytest.mark.parametrize(
    "vecdb", ["lancedb", "qdrant_cloud", "qdrant_local", "chroma"], indirect=True
)
@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
@pytest.mark.asyncio
async def test_doc_chat_agent_llm_async(
    test_settings: Settings, agent, query: str, expected: str
):
    """
    Test directly using `llm_response_async` method of DocChatAgent.
    """

    # note that the (query, ans) pairs are accumulated into the
    # internal dialog history of the agent.
    set_global(test_settings)
    agent.config.conversation_mode = False
    ans = (await agent.llm_response_async(query)).content
    expected = [e.strip() for e in expected.split(",")]
    assert all([e in ans for e in expected])


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
def test_doc_chat_agent_task(test_settings: Settings, agent, query, expected):
    """
    Test DocChatAgent wrapped in a Task.
    """
    set_global(test_settings)
    agent.config.conversation_mode = True
    task = Task(agent, restart=True)
    task.init()
    # LLM responds to Sys msg, initiates conv, says thank you, etc.
    task.step()

    agent.default_human_response = query
    task.step()  # user asks query
    task.step()  # LLM answers
    ans = task.pending_message.content.lower()
    expected = [e.strip() for e in expected.split(",")]
    assert all([e.lower() in ans for e in expected])
    assert task.pending_message.metadata.sender == Entity.LLM


class RetrievalAgent(DocChatAgent):
    def llm_response(
        self,
        message: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        # override the DocChatAgent's LLM response,
        # to just use ChatAgent's LLM response - this ensures that the system msg
        # is respected, and it uses the `retrieval_tool` as instructed.
        return ChatAgent.llm_response(self, message)


@pytest.fixture(scope="function")
def retrieval_agent(test_settings: Settings, vecdb) -> RetrievalAgent:
    set_global(test_settings)
    agent = RetrievalAgent(config)
    agent.vecdb = vecdb
    agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return agent


@pytest.mark.parametrize("vecdb", ["qdrant_local"], indirect=True)
@pytest.mark.parametrize(
    "query, expected",
    [
        (
            """
            Use the retrieval_tool to find out the capital of
            the fictional country NeoGlobal.
            """,
            "Winegarten",
        ),
        (
            "Use the retrieval_tool to answer this:" " who was Charlie Foster?",
            "comedian",
        ),
    ],
)
def test_retrieval_tool(
    test_settings: Settings, retrieval_agent, query: str, expected: str
):
    set_global(test_settings)
    retrieval_agent.enable_message(RetrievalTool)
    task = Task(
        retrieval_agent,
        restart=True,
        interactive=False,
        system_message=f"""
        To answer user's query, use the `retrieval_tool` to retrieve relevant passages, 
        and ONLY then answer the query. 
        In case the query is simply a topic or search phrase, 
        guess what the user may want to know, and formulate it as a 
        question to be answered, and use this as the `query` field in the 
        `retrieval_tool`. 
                
        When you are ready to show your answer, say {DONE}, followed by the answer.
        """,
    )
    # 3 turns:
    # 1. LLM gen `retrieval_tool` request
    # 2. Agent gen `retrieval_tool` response (i.e. returns relevant passages)
    # 3. LLM gen answer based on passages
    ans = task.run(query, turns=3).content
    expected = [e.strip().lower() for e in expected.split(",")]
    assert all([e in ans.lower() for e in expected])


@pytest.fixture(scope="function")
def new_agent(test_settings: Settings, vecdb) -> DocChatAgent:
    set_global(test_settings)
    agent = DocChatAgent(config)
    agent.vecdb = vecdb
    agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return agent


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
@pytest.mark.parametrize("conv_mode", [True, False])
@pytest.mark.parametrize("retain_context", [True, False])
def test_doc_chat_followup(
    test_settings: Settings, new_agent, conv_mode: bool, retain_context: bool
):
    """
    Test whether follow-up question is handled correctly.
    """
    new_agent.config.conversation_mode = conv_mode
    new_agent.config.retain_context = retain_context
    set_global(test_settings)
    task = Task(
        new_agent,
        interactive=False,
        restart=False,  # don't restart, so we can ask follow-up questions
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
    )
    result = task.run("Who was Charlie Foster?")
    assert "comedian" in result.content.lower()

    result = task.run("When was he born?")
    assert "1889" in result.content

    # test retain_context when conv_mode is True
    new_agent.init_state()
    question = "Who was Charlie Foster?"
    response = new_agent.llm_response(question)
    assert "comedian" in response.content.lower()
    if conv_mode:
        if retain_context:
            # context is retained, i.e.,
            # the user msg has both extracted chunks and the question itself
            assert len(new_agent.message_history[-2].content) > 2 * len(question)
        else:
            # context is not retained, i.e., the user msg has only the question
            assert len(new_agent.message_history[-2].content) < len(question) + 10


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
@pytest.mark.parametrize("conv_mode", [True, False])
@pytest.mark.parametrize("retain_context", [True, False])
@pytest.mark.asyncio
async def test_doc_chat_followup_async(
    test_settings: Settings,
    new_agent,
    conv_mode: bool,
    retain_context: bool,
):
    """
    Test whether follow-up question is handled correctly (in async mode).
    """
    new_agent.config.conversation_mode = conv_mode
    new_agent.config.retain_context = retain_context
    set_global(test_settings)
    task = Task(
        new_agent,
        interactive=False,
        restart=False,  # don't restart, so we can ask follow-up questions
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
    )
    result = await task.run_async("Who was Charlie Foster?")
    assert "comedian" in result.content.lower()

    result = await task.run_async("When was he born?")
    assert "1889" in result.content

    # test retain_context when conv_mode is True
    new_agent.init_state()
    question = "Who was Charlie Foster?"
    response = await new_agent.llm_response_async(question)
    assert "comedian" in response.content.lower()
    if conv_mode:
        if retain_context:
            # context is retained, i.e.,
            # the user msg has both extracted chunks and the question itself
            assert len(new_agent.message_history[-2].content) > 2 * len(question)
        else:
            # context is not retained, i.e., the user msg has only the question
            assert len(new_agent.message_history[-2].content) < len(question) + 10


# setup config for retrieval test, with n_neighbor_chunks=2
# and parser.n_neighbor_ids = 5
class _MyDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    n_neighbor_chunks = 2
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig | None = None

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )

    n_similar_chunks = 2
    n_relevant_chunks = 2
    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_neighbor_ids=5,
    )


@pytest.mark.parametrize("vecdb", ["lancedb", "chroma", "qdrant_local"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.PARA_SENTENCE, Splitter.SIMPLE, Splitter.TOKENS]
)
@pytest.mark.parametrize("conv_mode", [True, False])
def test_doc_chat_retrieval(
    test_settings: Settings, vecdb, splitter: Splitter, conv_mode: bool
):
    """
    Test window retrieval of relevant doc-chunks.
    Check that we are retrieving 2 neighbors around each match.
    """
    agent = DocChatAgent(
        _MyDocChatAgentConfig(
            n_similar_chunks=3,
            n_relevant_chunks=3,
            parsing=ParsingConfig(
                splitter=splitter,
            ),
        )
    )
    agent.config.conversation_mode = conv_mode
    agent.vecdb = vecdb

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
    agent.clear()
    agent.ingest_docs([Document(content=text, metadata={"source": "animals"})])
    results = agent.get_relevant_chunks("What are giraffes like?")

    # All phrases except the CATS phrase should be in the results
    # since they are all within 2 chunks of a giraffe phrase.
    # (The CAT phrase is 3 chunks away, so it should not be in the results.)
    all_but_cats = [p for p in vars(phrases).values() if "Cats" not in p]
    # check that each phrases occurs in exactly one result
    assert (
        sum(p in r.content for p in all_but_cats for r in results)
        == len(vars(phrases)) - 1
    )

    agent.clear()


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
def test_doc_chat_rerank_diversity(test_settings: Settings, vecdb):
    """
    Test that reranking by diversity works.
    """

    cfg = _MyDocChatAgentConfig(
        n_neighbor_chunks=0,
    )
    cfg.n_similar_chunks = 8
    cfg.n_relevant_chunks = 8
    agent = DocChatAgent(cfg)
    agent.vecdb = vecdb

    set_global(test_settings)

    phrases = SimpleNamespace(
        g1="Giraffes are tall.",
        g2="Giraffes are vegetarian.",
        g3="Giraffes are strange.",
        g4="Giraffes are fast.",
        g5="Giraffes are known to be tall.",
        g6="Giraffes are considered strange.",
        g7="Giraffes move fast.",
        g8="Giraffes are definitely vegetarian.",
    )
    docs = [
        Document(content=p, metadata=DocMetaData(source="user"))
        for p in vars(phrases).values()
    ]
    reranked = agent.rerank_with_diversity(docs)

    # assert that each phrase tall, vegetarian, strange, fast
    # occurs exactly once in top 4 phrases
    for p in ["tall", "vegetarian", "strange", "fast"]:
        assert sum(p in r.content for r in reranked[:4]) == 1


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
def test_reciprocal_rank_fusion(test_settings: Settings, vecdb):
    """
    Test that RRF (Reciprocal Rank Fusion) works.
    """

    cfg = _MyDocChatAgentConfig(
        n_neighbor_chunks=0,
        cross_encoder_reranking_model="",
        use_bm25_search=True,
        use_fuzzy_match=True,
        use_reciprocal_rank_fusion=True,
        parsing=ParsingConfig(
            splitter=Splitter.SIMPLE,
            n_neighbor_ids=5,
            n_similar_docs=None,  # Ensure we don't trigger backward compatibility
        ),
    )
    cfg.n_similar_chunks = 3
    cfg.n_relevant_chunks = 3
    agent = DocChatAgent(cfg)
    agent.vecdb = vecdb

    set_global(test_settings)

    phrases = SimpleNamespace(
        g1="time flies like an arrow",
        g2="a fly is very small",
        g3="we like apples",
        g4="the river bank got flooded",
        g5="there was a run on the bank",
        g6="JPMChase is a bank",
        g7="Chase is one of the banks",
    )
    docs = [
        Document(content=p, metadata=DocMetaData(source="user"))
        for p in vars(phrases).values()
    ]
    agent.ingest_docs(docs, split=False)
    chunks = agent.get_relevant_chunks("I like to chase banks")
    assert len(chunks) == 3
    assert any(phrases.g7 in chunk.content for chunk in chunks)
    assert any(phrases.g6 in chunk.content for chunk in chunks)

    chunks = agent.get_relevant_chunks("I like oranges")
    assert len(chunks) == 3
    assert any(phrases.g3 in chunk.content for chunk in chunks)
    assert any(phrases.g1 in chunk.content for chunk in chunks)


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma"], indirect=True)
def test_doc_chat_rerank_periphery(test_settings: Settings, vecdb):
    """
    Test that reranking to periphery works.
    """

    cfg = _MyDocChatAgentConfig(
        n_neighbor_chunks=0,
    )
    cfg.n_similar_chunks = 8
    cfg.n_relevant_chunks = 8
    agent = DocChatAgent(cfg)
    agent.vecdb = vecdb

    set_global(test_settings)

    docs = [
        Document(content=str(i), metadata=DocMetaData(source="user")) for i in range(10)
    ]
    reranked = agent.rerank_to_periphery(docs)
    numbers = [int(d.content) for d in reranked]
    assert numbers == [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]


data = {
    "id": ["A100", "B200", "C300", "D400", "E500"],
    "year": [1955, 1977, 1989, 2001, 2015],
    "author": [
        "Isaac Maximov",
        "J.K. Bowling",
        "George Morewell",
        "J.R.R. Bolshine",
        "Hugo Wellington",
    ],
    "title": [
        "The Last Question",
        "Harry Potter",
        "2084",
        "The Lord of the Rings",
        "The Time Machine",
    ],
    "summary": [
        "A story exploring the concept of entropy and the end of the universe.",
        "The adventures of a young wizard and his friends at a magical school.",
        "A dystopian novel about a totalitarian regime and the concept of freedom.",
        "An epic fantasy tale of a quest to destroy a powerful ring.",
        "A science fiction novel about time travel and its consequences.",
    ],
}

df = pd.DataFrame(data)


@pytest.mark.parametrize("metadata", [[], ["id", "year"], ["year"]])
@pytest.mark.parametrize("vecdb", ["lancedb", "qdrant_local", "chroma"], indirect=True)
def test_doc_chat_ingest_df(
    test_settings: Settings,
    vecdb,
    metadata,
):
    """Check we can ingest from a dataframe and run queries."""
    set_global(test_settings)

    sys_msg = "You will be asked to answer questions based on short book descriptions."
    agent_cfg = DocChatAgentConfig(
        system_message=sys_msg,
        cross_encoder_reranking_model="",
    )
    if isinstance(vecdb, LanceDB):
        agent = LanceDocChatAgent(agent_cfg)
    else:
        agent = DocChatAgent(agent_cfg)
    agent.vecdb = vecdb
    agent.clear()
    agent.ingest_dataframe(df, content="summary", metadata=metadata)
    response = agent.llm_response(
        """
        What concept does the book dealing with the end of the universe explore?
        """
    )
    assert "entropy" in response.content.lower()


@pytest.mark.parametrize("metadata", [[], ["id", "year"], ["year"]])
@pytest.mark.parametrize("vecdb", ["lancedb", "qdrant_local", "chroma"], indirect=True)
def test_doc_chat_add_content_fields(
    test_settings: Settings,
    vecdb,
    metadata,
):
    """Check we can ingest from a dataframe,
    with additional fields inserted into content,
    and run queries that refer to those fields."""

    set_global(test_settings)

    sys_msg = "You will be asked to answer questions based on short movie descriptions."
    agent_cfg = DocChatAgentConfig(
        system_message=sys_msg,
        cross_encoder_reranking_model="",
        add_fields_to_content=["year", "author", "title"],
    )
    if isinstance(vecdb, LanceDB):
        agent = LanceDocChatAgent(agent_cfg)
    else:
        agent = DocChatAgent(agent_cfg)
    agent.vecdb = vecdb
    agent.clear()
    agent.ingest_dataframe(df, content="summary", metadata=metadata)
    response = agent.llm_response(
        """
        What was the title of the book by George Morewell and when was it written?
        """
    )
    assert "2084" in response.content and "1989" in response.content


@pytest.mark.parametrize("vecdb", ["lancedb", "chroma", "qdrant_local"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.PARA_SENTENCE, Splitter.SIMPLE, Splitter.TOKENS]
)
def test_doc_chat_incremental_ingest(
    test_settings: Settings, vecdb, splitter: Splitter
):
    """
    Check that we are able ingest documents incrementally.
    """
    agent = DocChatAgent(
        _MyDocChatAgentConfig(
            n_similar_chunks=3,
            n_relevant_chunks=3,
            parsing=ParsingConfig(
                splitter=splitter,
            ),
        )
    )
    agent.vecdb = vecdb

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
    sentences = list(vars(phrases).values())
    docs1 = [
        Document(content=s, metadata=dict(source="animals")) for s in sentences[:4]
    ]

    docs2 = [
        Document(content=s, metadata=dict(source="animals")) for s in sentences[4:]
    ]
    agent.ingest_docs(docs1)
    assert agent.vecdb.config.collection_name in agent.vecdb.list_collections(True)
    agent.ingest_docs(docs2)
    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)

    results = agent.get_relevant_chunks("What do we know about Hyenas?")
    assert any("fast" in r.content for r in results) or any(
        "dangerous" in r.content for r in results
    )


@pytest.mark.parametrize("vecdb", ["chroma", "qdrant_local"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.PARA_SENTENCE, Splitter.SIMPLE, Splitter.TOKENS]
)
@pytest.mark.parametrize("source", ["bytes", "path"])
def test_doc_chat_ingest_paths(
    test_settings: Settings,
    vecdb,
    splitter: Splitter,
    source,
):
    """
    Test DocChatAgent.ingest_doc_paths
    """
    agent = DocChatAgent(
        _MyDocChatAgentConfig(
            n_similar_chunks=3,
            n_relevant_chunks=3,
            parsing=ParsingConfig(
                splitter=splitter,
            ),
        )
    )
    agent.vecdb = vecdb

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
    sentences = list(vars(phrases).values())

    # create temp files containing each sentence, using tempfile pkg
    import tempfile

    for s in sentences:
        if source == "path":
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(s)
                f.close()
                agent.ingest_doc_paths([f.name])
        else:
            agent.ingest_doc_paths([s.encode()])

    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)

    results = agent.get_relevant_chunks("What do we know about Hyenas?")
    assert any("fast" in r.content for r in results) or any(
        "dangerous" in r.content for r in results
    )


@pytest.mark.xfail(
    condition=lambda: "lancedb" in vecdb,
    reason="LanceDB may fail due to unknown flakiness",
    run=True,
    strict=False,
)
@pytest.mark.parametrize("vecdb", ["chroma", "lancedb", "qdrant_local"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.PARA_SENTENCE, Splitter.SIMPLE, Splitter.TOKENS]
)
@pytest.mark.parametrize("metadata_dict", [True, False])
def test_doc_chat_ingest_path_metadata(
    test_settings: Settings,
    vecdb,
    splitter: Splitter,
    metadata_dict: bool,  # whether metadata is dict or DocMetaData
):
    """
    Test DocChatAgent.ingest_doc_paths, with metadata
    """
    agent = DocChatAgent(
        _MyDocChatAgentConfig(
            n_similar_chunks=3,
            n_relevant_chunks=3,
            parsing=ParsingConfig(
                splitter=splitter,
            ),
        )
    )
    agent.vecdb = vecdb

    set_global(test_settings)

    # create a list of dicts, each containing a sentence about an animal
    # and a metadata field indicating the animal's name, species, and diet
    animals = [
        {
            "content": "Cats are quiet and clean.",
            "metadata": {
                "name": "cat",
                "species": "feline",
                "diet": "carnivore",
            },
        },
        {
            "content": "Dogs are loud and messy.",
            "metadata": {
                "name": "dog",
                "species": "canine",
                "diet": "omnivore",
            },
        },
        {
            "content": "Pigs cannot fly.",
            "metadata": {
                "name": "pig",
                "species": "porcine",
                "diet": "omnivore",
            },
        },
    ]

    class AnimalMetadata(DocMetaData):
        name: str
        species: str
        diet: str

    animal_metadata_list = [AnimalMetadata(**a["metadata"]) for a in animals]

    # put each animal content in a separate file
    import tempfile

    for animal in animals:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(animal["content"])
            f.close()
            animal["path"] = f.name

    agent.clear()
    # ingest with per-file metadata
    agent.ingest_doc_paths(
        [a["path"] for a in animals],
        metadata=(
            [a["metadata"] for a in animals] if metadata_dict else animal_metadata_list
        ),
    )
    assert agent.vecdb.config.collection_name in agent.vecdb.list_collections(True)

    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)
    # assert about metadata
    assert any("porcine" in r.metadata.species for r in results)

    # clear out the agent docs and the underlying vecdb collection
    agent.clear()

    # ingest with single metadata for ALL animals
    agent.ingest_doc_paths(
        [a["path"] for a in animals],
        metadata=(
            dict(type="animal", category="living")
            if metadata_dict
            else DocMetaData(type="animal", category="living")
        ),
    )
    assert agent.vecdb.config.collection_name in agent.vecdb.list_collections(True)

    results = agent.get_relevant_chunks("What do we know about dogs?")
    assert any("messy" in r.content for r in results)
    assert all(r.metadata.type == "animal" for r in results)

    agent.clear()


@pytest.mark.xfail(
    condition=lambda: "lancedb" in vecdb,
    reason="LanceDB may fail due to unknown flakiness",
    run=True,
    strict=False,
)
@pytest.mark.parametrize("vecdb", ["chroma", "lancedb", "qdrant_local"], indirect=True)
def test_doc_chat_batch(test_settings: Settings, vecdb):
    """
    Test batch run of queries to multiple instances of DocChatAgent,
    which share the same vector-db.
    """

    set_global(test_settings)
    doc_agents = [DocChatAgent(_MyDocChatAgentConfig()) for _ in range(2)]

    # attach a common vector-db to all agents
    for a in doc_agents:
        a.vecdb = vecdb

    docs = [
        Document(
            content="""
            Filidor Dinkoyevsky wrote a book called "The Sisters Karenina".
            It is loosely based on the life of the Anya Karvenina,
            from a book by Tolsitoy a few years earlier.
            """,
            metadata=DocMetaData(source="tweakipedia"),
        ),
        Document(
            content="""
            The novel "Searching for Sebastian Night" was written by Vlad Nabikov.
            It is an intriguing tale about the author's search for his lost brother,
            and is a meditation on the nature of loss and memory.
            """,
            metadata=DocMetaData(source="tweakipedia"),
        ),
    ]

    # note we only need to ingest docs using one of the agents,
    # since they share the same vector-db
    doc_agents[0].ingest_docs(docs, split=False)

    questions = [
        "What book did Vlad Nabikov write?",
        "Who wrote the book about the Karenina sisters?",
    ]

    # (1) test that we can create a single task and use run_batch_tasks
    task = Task(doc_agents[0], name="DocAgent", interactive=False, single_round=True)
    results = run_batch_tasks(task, questions)

    assert "Sebastian" in results[0].content
    assert "Dinkoyevsky" in results[1].content

    # (2) test that we can create a task-generator fn and use run_batch_task_gen

    # create a task-generator fn, to create one per question
    def gen_task(i: int):
        return Task(
            doc_agents[i],
            name=f"DocAgent-{i}",
            interactive=False,
            single_round=True,
        )

    results = run_batch_task_gen(gen_task, questions)

    assert "Sebastian" in results[0].content
    assert "Dinkoyevsky" in results[1].content

    for a in doc_agents:
        a.clear()


@pytest.mark.parametrize("enrichment, expect_cleaned", [(True, True), (False, False)])
def test_remove_enrichments(
    enrichment: bool,
    expect_cleaned: bool,
) -> None:
    """
    Test removal of generated enrichments from documents both if
    they have enrichment or not.
    """

    original_content = "This is the original content"
    sample_docs = [
        Document(
            content=f"""
                {original_content}\n\n{CHUNK_ENRICHMENT_DELIMITER}
                Some generated enrichments
            """,
            metadata=DocMetaData(source="one", has_enrichment=True),
        ),
        Document(
            content="This is a normal document", metadata=DocMetaData(source="twp")
        ),
    ]

    enrichment_config = ChunkEnrichmentAgentConfig() if enrichment else None
    agent = DocChatAgent(
        _TestDocChatAgentConfig(chunk_enrichment_config=enrichment_config)
    )

    cleaned_docs = agent.remove_chunk_enrichments(sample_docs)

    assert len(cleaned_docs) == len(sample_docs)
    # Document with questions should be cleaned
    assert (cleaned_docs[0].content.strip() == original_content) is expect_cleaned
    # Normal document should be unchanged
    assert cleaned_docs[1].content == sample_docs[1].content


def test_add_enrichments() -> None:
    """Test generation of enrichments for documents"""

    sample_docs = [
        Document(content="Doc 1", metadata=DocMetaData(source="one")),
        Document(content="Doc 2", metadata=DocMetaData(source="two")),
    ]

    enrichment_agent_config = ChunkEnrichmentAgentConfig(
        llm=MockLMConfig(
            response_fn=lambda x: "Keyword1, keyword2 " + x,
        ),
        enrichment_prompt_fn=lambda x: "Add keywords to " + x,
    )
    agent = DocChatAgent(
        _TestDocChatAgentConfig(
            chunk_enrichment_config=enrichment_agent_config,
        )
    )

    augmented_docs = agent.enrich_chunks(sample_docs)

    assert len(augmented_docs) == len(sample_docs)
    for doc in augmented_docs:
        assert CHUNK_ENRICHMENT_DELIMITER in doc.content
        assert doc.metadata.has_enrichment
        # Original content should be preserved before marker
        assert (
            doc.content.split(CHUNK_ENRICHMENT_DELIMITER)[0].strip()
            in sample_docs[0].content + sample_docs[1].content
        )
        # Should have enrichment after marker
        enrichment_part = doc.content.split(CHUNK_ENRICHMENT_DELIMITER)[1].strip()
        assert len(enrichment_part) > 0


def test_enrichments_disabled() -> None:
    """Test that enrichments are not generated when disabled"""

    sample_docs = [Document(content="Doc 1", metadata=DocMetaData(source="one"))]

    agent = DocChatAgent(_TestDocChatAgentConfig(chunk_enrichment_config=None))

    processed_docs = agent.enrich_chunks(sample_docs)
    assert len(processed_docs) == len(sample_docs)
    assert processed_docs[0].content == sample_docs[0].content
    assert not hasattr(processed_docs[0].metadata, "has_enrichment")

    cleaned_docs = agent.remove_chunk_enrichments(sample_docs)
    assert len(cleaned_docs) == len(sample_docs)
    assert cleaned_docs[0].content == sample_docs[0].content


@pytest.mark.parametrize(
    "vecdb",
    ["lancedb", "qdrant_local", "qdrant_cloud", "chroma"],
    indirect=True,
)
def test_enrichments_integration(vecdb: VectorStore) -> None:
    """Integration test for chunk-enrichments in RAG pipeline"""

    sample_docs = [
        Document(
            content="Blood Test name: BUN",
            # Blood Urea Nitrogen test for kidney function
            metadata=DocMetaData(source="blood-work", is_chunk=True),
        ),
        Document(
            content="Blood test name: BNP",
            # B-type natriuretic peptide test for heart function
            metadata=DocMetaData(source="blood-work", is_chunk=True),
        ),
    ]

    # add 20 random docs
    sample_docs.extend(
        [
            Document(
                content=generate_random_text(5),
                metadata=DocMetaData(source="blood-work", is_chunk=True),
            )
            for _ in range(20)
        ]
    )

    agent = DocChatAgent(
        _TestDocChatAgentConfig(
            rerank_diversity=False,
            rerank_periphery=False,
            relevance_extractor_config=None,
            conversation_mode=False,
            chunk_enrichment_config=ChunkEnrichmentAgentConfig(
                system_message="""
                You are an expert in medical tests, well-versed in 
                test names and which organs they are associated with.
                """,
                enrichment_prompt_fn=lambda x: (
                    f"""
                    Which organ function or health is the following blood test
                    most closely associated with? 
                    (Answer in ONE WORD; if unsure, say UNKNOWN)
                    
                    {x}
                    """
                ),
            ),
        )
    )
    agent.vecdb = vecdb
    agent.augment_system_message(
        """
        
        You are an expert in medical tests, well-versed in 
        test names and which organs they are associated with.
        
        You must answer questions based on the provided document-extracts,
        combined with your medical knowledge.
        """
    )
    # clear existing docs
    agent.clear()
    agent.ingest_docs(sample_docs)

    # Verify questions were generated during ingestion
    all_docs = agent.vecdb.get_all_documents()
    assert all(
        CHUNK_ENRICHMENT_DELIMITER in doc.content for doc in all_docs
    ), "Generated enrichments not found in all docs"

    # retrieve docs they should not contain generated enrichments
    doc1 = agent.answer_from_docs("Which medical test is kidney-related?")
    doc2 = agent.answer_from_docs("Which blood test is related to heart-function?")

    # they are documents enriched with keywords
    # but the keywords should be removed on retrieval
    assert (
        CHUNK_ENRICHMENT_DELIMITER not in doc1.content
    ), f"Doc 1 has not been cleaned: {doc1.content}"
    assert (
        CHUNK_ENRICHMENT_DELIMITER not in doc2.content
    ), f"Doc 2 has not been cleaned: {doc2.content}"

    # check that the right content is in the docs
    assert "BUN" in doc1.content
    assert "BNP" in doc2.content


@pytest.mark.parametrize("vecdb", ["chroma", "lancedb", "qdrant_local"], indirect=True)
def test_doc_chat_agent_ingest(test_settings: Settings, vecdb):
    agent = DocChatAgent(_MyDocChatAgentConfig())
    agent.vecdb = vecdb
    agent.clear()  # clear the collection in the vector store

    # Base documents with simple metadata
    docs = [
        Document(content="Doc 1", metadata=DocMetaData(source="original1")),
        Document(content="Doc 2", metadata=DocMetaData(source="original2")),
    ]

    # Test case 1: List of metadata dicts
    docs_copy = [d.model_copy() for d in docs]
    meta_list = [
        {"category": "A", "source": "new1"},
        {"category": "B", "source": "new2"},
    ]
    agent.ingest_docs(docs_copy, metadata=meta_list)
    stored = agent.vecdb.get_all_documents()
    assert set([d.metadata.category for d in stored]) == {"A", "B"}

    assert set(d.metadata.source for d in stored) == {
        _append_metadata_source("original1", "new1"),
        _append_metadata_source("original2", "new2"),
    }

    agent.clear()

    # Test case 2: Single metadata dict for all docs
    docs_copy = [d.model_copy() for d in docs]
    meta_dict = {"category": "common", "source": "new"}
    agent.ingest_docs(docs_copy, metadata=meta_dict)
    stored = agent.vecdb.get_all_documents()
    assert all(d.metadata.category == "common" for d in stored)

    assert set(d.metadata.source for d in stored) == {
        _append_metadata_source("original1", "new"),
        _append_metadata_source("original2", "new"),
    }
    agent.clear()

    # Test case 3: List of DocMetaData
    docs_copy = [d.model_copy() for d in docs]
    meta_docs = [
        DocMetaData(category="X", source="new1"),
        DocMetaData(category="Y", source="new2"),
    ]
    agent.ingest_docs(docs_copy, metadata=meta_docs)
    stored = agent.vecdb.get_all_documents()
    assert set([d.metadata.category for d in stored]) == {"X", "Y"}

    assert set(d.metadata.source for d in stored) == {
        _append_metadata_source("original1", "new1"),
        _append_metadata_source("original2", "new2"),
    }
    agent.clear()

    # Test case 4: Single DocMetaData for all docs
    docs_copy = [d.model_copy() for d in docs]
    meta_doc = DocMetaData(category="shared", source="new")
    agent.ingest_docs(docs_copy, metadata=meta_doc)
    stored = agent.vecdb.get_all_documents()
    assert all(d.metadata.category == "shared" for d in stored)

    assert set(d.metadata.source for d in stored) == {
        _append_metadata_source("original1", "new"),
        _append_metadata_source("original2", "new"),
    }
    agent.clear()


@pytest.mark.parametrize("vecdb", ["chroma", "lancedb", "qdrant_local"], indirect=True)
def test_doc_chat_agent_ingest_paths(test_settings: Settings, vecdb):
    agent = DocChatAgent(_MyDocChatAgentConfig())
    agent.vecdb = vecdb
    agent.clear()  # clear the collection in the vector store

    # Create temp files and byte contents
    import tempfile

    # Create two temp files
    file_contents = ["Content of file 1", "Content of file 2"]
    temp_files = []
    for content in file_contents:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_files.append(f.name)

    # Create two byte contents
    byte_contents = [b"Bytes content 1", b"Bytes content 2"]

    # Test case 1: List of metadata dicts
    paths = temp_files + byte_contents
    meta_list = [
        {"category": "file", "source": "src1"},
        {"category": "file", "source": "src2"},
        {"category": "bytes", "source": "src3"},
        {"category": "bytes", "source": "src4"},
    ]
    # ingest with no additional metadata so we can get the original source
    docs = agent.ingest_doc_paths(paths)
    orig_sources = [d.metadata.source for d in docs]
    agent.clear()
    # now ingest with additional metadata
    agent.ingest_doc_paths(paths, metadata=meta_list)
    stored = agent.vecdb.get_all_documents()

    # Create sets of expected and actual metadata
    expected_categories = {"file", "bytes"}
    expected_sources = {
        _append_metadata_source(s, f"src{i+1}") for i, s in enumerate(orig_sources)
    }

    actual_categories = {d.metadata.category for d in stored}
    actual_sources = {d.metadata.source for d in stored}

    assert expected_categories == actual_categories
    assert expected_sources == actual_sources
    agent.clear()

    # Test case 2: Single metadata dict
    meta_dict = {"category": "common", "source": "shared"}
    # now ingest with additional metadata
    agent.ingest_doc_paths(paths, metadata=meta_dict)
    stored = agent.vecdb.get_all_documents()

    assert all(d.metadata.category == "common" for d in stored)
    expected_sources = {_append_metadata_source(s, "shared") for s in orig_sources}
    actual_sources = {d.metadata.source for d in stored}
    assert expected_sources == actual_sources
    agent.clear()

    # Test case 3: List of DocMetaData
    meta_docs = [DocMetaData(category="X", source=f"src{i}") for i in range(len(paths))]

    # now ingest with metadata
    agent.ingest_doc_paths(paths, metadata=meta_docs)
    stored = agent.vecdb.get_all_documents()

    expected_categories = {"X"}
    expected_sources = {
        _append_metadata_source(s, f"src{i}") for i, s in enumerate(orig_sources)
    }

    actual_categories = {d.metadata.category for d in stored}
    actual_sources = {d.metadata.source for d in stored}

    assert expected_categories == actual_categories
    assert expected_sources == actual_sources
    agent.clear()

    # Test case 4: Single DocMetaData
    meta_doc = DocMetaData(category="shared", source="new")
    docs = agent.ingest_doc_paths(paths, metadata=meta_doc)
    stored = agent.vecdb.get_all_documents()

    assert all(d.metadata.category == "shared" for d in stored)
    expected_sources = {_append_metadata_source(s, "new") for s in orig_sources}
    actual_sources = {d.metadata.source for d in stored}
    assert expected_sources == actual_sources
    agent.clear()

    # Cleanup temp files
    for f in temp_files:
        os.remove(f)
