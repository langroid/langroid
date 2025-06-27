import warnings
from types import SimpleNamespace

import pytest

from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import ParsingConfig, Splitter
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


set_global(Settings(cache=True))  # allow cacheing


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)


# setup config for retrieval test, with n_neighbor_chunks=2
# and parser.n_neighbor_ids = 5
class _MyDocChatAgentConfig(DocChatAgentConfig):
    cross_encoder_reranking_model = ""
    n_query_rephrases = 0
    n_neighbor_chunks = 2
    n_similar_chunks = 2
    n_relevant_chunks = 2
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        collection_name="test-data",
        replace_collection=True,
        storage_path=":memory:",
        embedding=OpenAIEmbeddingsConfig(
            model_name="text-embedding-3-small",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )

    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_neighbor_ids=5,
    )


@pytest.mark.parametrize("vecdb", ["chroma", "qdrant_local", "lancedb"], indirect=True)
@pytest.mark.parametrize(
    "splitter", [Splitter.SIMPLE, Splitter.PARA_SENTENCE, Splitter.TOKENS]
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
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


@pytest.mark.parametrize("vecdb", ["qdrant_local", "chroma", "lancedb"], indirect=True)
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
