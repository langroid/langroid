import os
import warnings
from types import SimpleNamespace
from typing import List

import pandas as pd
import pytest

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
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

for _ in range(100):
    documents.append(
        Document(
            content=generate_random_text(5),
            metadata=DocMetaData(source="random"),
        )
    )


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


class _TestDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    # vecdb: VectorStoreConfig = QdrantDBConfig(
    #     collection_name="test-data",
    #     replace_collection=True,
    #     storage_path=storage_path,
    #     embedding=OpenAIEmbeddingsConfig(
    #         model_type="openai",
    #         model_name="text-embedding-ada-002",
    #         dims=1536,
    #     ),
    # )

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


@pytest.fixture(scope="function")
def agent(vecdb) -> DocChatAgent:
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

QUERY_EXPECTED_PAIRS = [
    ("what happened in the year 2050?", "GPT10, Lithuania"),
    ("what is the capital of England?", "Paris"),
    ("Who was Charlie Chaplin?", "comedian"),
    ("What used to be capital of France?", "Paris"),
    ("When was global warming solved?", "2060"),
    ("What do we know about paperclips?", "2057, 2061"),
]


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
@pytest.mark.parametrize("conv_mode", [True, False])
def test_doc_chat_followup(test_settings: Settings, agent, conv_mode: bool):
    """
    Test whether follow-up question is handled correctly.
    """
    agent.config.conversation_mode = conv_mode
    set_global(test_settings)
    task = Task(
        agent,
        interactive=False,
        restart=True,
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
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
        storage_path=":memory:",
        embedding=OpenAIEmbeddingsConfig(
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


@pytest.mark.parametrize("vecdb", ["chroma", "qdrant_local", "lancedb"], indirect=True)
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
            parsing=ParsingConfig(
                splitter=splitter,
                n_similar_docs=3,
            )
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
def test_doc_chat_rerank_diversity(test_settings: Settings, vecdb):
    """
    Test that reranking by diversity works.
    """

    cfg = _MyDocChatAgentConfig(
        n_neighbor_chunks=0,
    )
    cfg.parsing.n_similar_docs = 8
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
def test_doc_chat_rerank_periphery(test_settings: Settings, vecdb):
    """
    Test that reranking to periphery works.
    """

    cfg = _MyDocChatAgentConfig(
        n_neighbor_chunks=0,
    )
    cfg.parsing.n_similar_docs = 8
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
    agent.ingest_dataframe(df, content="summary", metadata=metadata)
    response = agent.llm_response(
        """
        What was the title of the George Morewell book and when was it written?
        """
    )
    assert "2084" in response.content and "1989" in response.content


@pytest.mark.parametrize("vecdb", ["chroma", "qdrant_local", "lancedb"], indirect=True)
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
            parsing=ParsingConfig(
                splitter=splitter,
                n_similar_docs=3,
            )
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
    agent.ingest_docs(docs2)
    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)

    results = agent.get_relevant_chunks("What do we know about Hyenas?")
    assert any("fast" in r.content for r in results) or any(
        "dangerous" in r.content for r in results
    )


@pytest.mark.parametrize("vecdb", ["chroma", "qdrant_local", "lancedb"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.PARA_SENTENCE, Splitter.SIMPLE, Splitter.TOKENS]
)
def test_doc_chat_ingest_paths(test_settings: Settings, vecdb, splitter: Splitter):
    """
    Test DocChatAgent.ingest_doc_paths
    """
    agent = DocChatAgent(
        _MyDocChatAgentConfig(
            parsing=ParsingConfig(
                splitter=splitter,
                n_similar_docs=3,
            )
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
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(s)
            f.close()
            agent.ingest_doc_paths([f.name])

    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)

    results = agent.get_relevant_chunks("What do we know about Hyenas?")
    assert any("fast" in r.content for r in results) or any(
        "dangerous" in r.content for r in results
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
            parsing=ParsingConfig(
                splitter=splitter,
                n_similar_docs=3,
            )
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

    # ingest with per-file metadata
    agent.ingest_doc_paths(
        [a["path"] for a in animals],
        metadata=[a["metadata"] for a in animals]
        if metadata_dict
        else animal_metadata_list,
    )

    results = agent.get_relevant_chunks("What do we know about Pigs?")
    assert any("fly" in r.content for r in results)
    # assert about metadata
    assert any("porcine" in r.metadata.species for r in results)

    # clear out the agent docs and the underlying vecdb collection
    agent.clear()

    # ingest with single metadata for ALL animals
    agent.ingest_doc_paths(
        [a["path"] for a in animals],
        metadata=dict(type="animal", category="living")
        if metadata_dict
        else DocMetaData(type="animal", category="living"),
    )

    results = agent.get_relevant_chunks("What do we know about dogs?")
    assert any("messy" in r.content for r in results)
    assert all(r.metadata.type == "animal" for r in results)

    agent.clear()
