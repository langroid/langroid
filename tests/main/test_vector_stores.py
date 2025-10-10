import json
from types import SimpleNamespace
from typing import List

import pytest
from dotenv import load_dotenv
from sqlalchemy.exc import OperationalError

from langroid.agent.batch import run_batch_tasks
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.task import Task
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import Parser, ParsingConfig, Splitter
from langroid.utils.system import rmdir
from langroid.vector_store.base import VectorStore
from langroid.vector_store.lancedb import LanceDB, LanceDBConfig
from langroid.vector_store.meilisearch import MeiliSearch, MeiliSearchConfig
from langroid.vector_store.pineconedb import PineconeDB, PineconeDBConfig
from langroid.vector_store.postgres import PostgresDB, PostgresDBConfig
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
from langroid.vector_store.weaviatedb import WeaviateDB, WeaviateDBConfig

load_dotenv()
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

phrases = SimpleNamespace(
    HELLO="hello",
    HI_THERE="hi there",
    CANADA="people living in Canada",
    NOT_CANADA="people not living in Canada",
    OVER_40="people over 40",
    UNDER_40="people under 40",
    FRANCE="what is the capital of France?",
    BELGIUM="which city is Belgium's capital?",
)


class MyDocMetaData(DocMetaData):
    id: str


class MyDoc(Document):
    content: str
    metadata: MyDocMetaData


stored_docs = [
    MyDoc(content=d, metadata=MyDocMetaData(id=str(i)))
    for i, d in enumerate(vars(phrases).values())
]


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
        qd.add_documents(stored_docs)
        yield qd
        return

    if request.param == "qdrant_cloud":
        qd_dir = ".qdrant/cloud/" + embed_cfg.model_type
        qd_cfg_cloud = QdrantDBConfig(
            cloud=True,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=qd_dir,
            embedding=embed_cfg,
        )
        qd_cloud = QdrantDB(qd_cfg_cloud)
        qd_cloud.add_documents(stored_docs)
        yield qd_cloud
        qd_cloud.delete_collection(collection_name=qd_cfg_cloud.collection_name)
        return
    if request.param == "weaviate_cloud":
        wv_cfg_cloud = WeaviateDBConfig(
            collection_name="test_" + embed_cfg.model_type,
            embedding=embed_cfg,
            cloud=True,
        )
        weaviate_cloud = WeaviateDB(wv_cfg_cloud)
        weaviate_cloud.add_documents(stored_docs)
        yield weaviate_cloud
        weaviate_cloud.delete_collection(collection_name=wv_cfg_cloud.collection_name)
        return
    if request.param == "weaviate_local":
        wv_dir = ".weaviate/" + embed_cfg.model_type
        rmdir(wv_dir)

        wv_cfg_local = WeaviateDBConfig(
            collection_name="test_" + embed_cfg.model_type,
            embedding=embed_cfg,
            cloud=False,
            docker=False,
            storage_path=wv_dir,
        )
        weaviate_local = WeaviateDB(wv_cfg_local)
        weaviate_local.add_documents(stored_docs)
        yield weaviate_local
        weaviate_local.delete_collection(collection_name=wv_cfg_local.collection_name)
        rmdir(wv_dir)
        return
    if request.param == "weaviate_docker":
        wv_cfg_docker = WeaviateDBConfig(
            collection_name="test_" + embed_cfg.model_type,
            embedding=embed_cfg,
            docker=True,
        )
        weaviate_docker = WeaviateDB(wv_cfg_docker)
        weaviate_docker.add_documents(stored_docs)
        yield weaviate_docker
        weaviate_docker.delete_collection(collection_name=wv_cfg_docker.collection_name)
        return

    if request.param == "qdrant_hybrid_cloud":
        qd_dir = ".qdrant/cloud/" + embed_cfg.model_type
        qd_cfg_cloud = QdrantDBConfig(
            cloud=True,
            collection_name="test-" + embed_cfg.model_type,
            replace_collection=True,
            storage_path=qd_dir,
            embedding=embed_cfg,
            use_sparse_embeddings=True,
            sparse_embedding_model="naver/splade-v3-distilbert",
        )
        qd_cloud = QdrantDB(qd_cfg_cloud)
        qd_cloud.add_documents(stored_docs)
        yield qd_cloud
        qd_cloud.delete_collection(collection_name=qd_cfg_cloud.collection_name)
        return

    if request.param == "chroma":
        try:
            from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig
        except ImportError:
            pytest.skip("Chroma not installed")
            return
        cd_dir = ".chroma/" + embed_cfg.model_type
        rmdir(cd_dir)
        cd_cfg = ChromaDBConfig(
            collection_name="test-" + embed_cfg.model_type,
            storage_path=cd_dir,
            embedding=embed_cfg,
        )
        cd = ChromaDB(cd_cfg)
        cd.add_documents(stored_docs)
        yield cd
        rmdir(cd_dir)
        return

    if request.param == "postgres":
        pg_cfg = PostgresDBConfig(
            collection_name="test_" + embed_cfg.model_type,
            embedding=embed_cfg,
            cloud=False,
            replace_collection=True,
        )
        pg = PostgresDB(pg_cfg)
        pg.add_documents(stored_docs)
        yield pg
        pg.delete_collection(collection_name=pg_cfg.collection_name)
        return

    if request.param == "meilisearch":
        ms_cfg = MeiliSearchConfig(
            collection_name="test-meilisearch",
        )
        ms = MeiliSearch(ms_cfg)
        ms.add_documents(stored_docs)
        yield ms
        ms.delete_collection(collection_name=ms_cfg.collection_name)
        return

    if request.param == "lancedb":
        ldb_dir = ".lancedb/data/" + embed_cfg.model_type
        rmdir(ldb_dir)
        ldb_cfg = LanceDBConfig(
            cloud=False,
            collection_name="test-" + embed_cfg.model_type,
            storage_path=ldb_dir,
            embedding=embed_cfg,
            # document_class=MyDoc,  # IMPORTANT, to ensure table has full schema!
        )
        ldb = LanceDB(ldb_cfg)
        ldb.add_documents(stored_docs)
        yield ldb
        rmdir(ldb_dir)
        return

    if request.param == "pinecone_serverless":
        cfg = PineconeDBConfig(
            collection_name="pinecone-serverless-test",
            embedding=embed_cfg,
        )
        pinecone_serverless = PineconeDB(config=cfg)
        pinecone_serverless.add_documents(stored_docs)
        yield pinecone_serverless
        pinecone_serverless.delete_collection(collection_name=cfg.collection_name)
        return


@pytest.mark.parametrize(
    "query,results,exceptions",
    [
        ("which city is Belgium's capital?", [phrases.BELGIUM], ["meilisearch"]),
        ("capital of France", [phrases.FRANCE], ["meilisearch"]),
        ("hello", [phrases.HELLO], ["meilisearch"]),
        ("hi there", [phrases.HI_THERE], ["meilisearch"]),
        ("men and women over 40", [phrases.OVER_40], ["meilisearch"]),
        ("people aged less than 40", [phrases.UNDER_40], ["meilisearch"]),
        ("Canadian residents", [phrases.CANADA], ["meilisearch"]),
        ("people outside Canada", [phrases.NOT_CANADA], ["meilisearch"]),
    ],
)
@pytest.mark.parametrize(
    "vecdb",
    [
        "weaviate_docker",
        "postgres",
        "qdrant_cloud",
        "qdrant_local",
        pytest.param("pinecone_serverless", marks=pytest.mark.skip),
        "lancedb",
        "chroma",
    ],
    indirect=True,
)
def test_vector_stores_search(
    vecdb, query: str, results: List[str], exceptions: List[str]
):
    if vecdb.__class__.__name__.lower() in exceptions:
        # we don't expect some of these to work,
        # e.g. MeiliSearch is a text search engine, not a vector store
        return
    assert vecdb.config.collection_name in vecdb.list_collections(True)
    docs_and_scores = vecdb.similar_texts_with_scores(query, k=len(vars(phrases)))
    # first doc should be best match
    # scores are cosine similarities, so high means close
    matching_docs = [doc.content for doc, score in docs_and_scores if score > 0.7]
    assert set(results).issubset(set(matching_docs))


@pytest.mark.xfail(
    reason="QdrantDB may fail saying `not ready`",
    run=True,
    strict=False,
)
@pytest.mark.parametrize(
    "query,results,exceptions",
    [
        ("which city is Belgium's capital?", [phrases.BELGIUM], ["meilisearch"]),
        ("capital of France", [phrases.FRANCE], ["meilisearch"]),
        ("hello", [phrases.HELLO], ["meilisearch"]),
        ("hi there", [phrases.HI_THERE], ["meilisearch"]),
        ("men and women over 40", [phrases.OVER_40], ["meilisearch"]),
        ("people aged less than 40", [phrases.UNDER_40], ["meilisearch"]),
        ("Canadian residents", [phrases.CANADA], ["meilisearch"]),
        ("people outside Canada", [phrases.NOT_CANADA], ["meilisearch"]),
    ],
)
@pytest.mark.parametrize(
    "vecdb",
    ["qdrant_hybrid_cloud"],
    indirect=True,
)
def test_hybrid_vector_search(
    vecdb, query: str, results: List[str], exceptions: List[str]
):
    if vecdb.__class__.__name__.lower() in exceptions:
        return
    docs_and_scores = vecdb.similar_texts_with_scores(query, k=len(vars(phrases)))
    # first doc should be best match
    # scores are cosine similarities, so high means close
    matching_docs = [doc.content for doc, score in docs_and_scores if score > 0.7]
    assert set(results).issubset(set(matching_docs))


@pytest.mark.parametrize(
    "vecdb",
    [
        "postgres",
        "lancedb",
        "chroma",
        "qdrant_local",
        "qdrant_cloud",
        pytest.param("pinecone_serverless", marks=pytest.mark.skip),
        "weaviate_docker",
    ],
    indirect=True,
)
def test_vector_stores_access(vecdb):
    assert vecdb is not None

    # test that we can ingest docs that are created
    # via subclass of Document and  DocMetaData.
    class MyDocMeta(DocMetaData):
        category: str  # an extra field

    class MyDocument(Document):
        content: str
        metadata: MyDocMeta

    vecdb.config.document_class = MyDocument
    vecdb.config.metadata_class = MyDocMeta
    coll_name = vecdb.config.collection_name
    assert coll_name is not None

    vecdb.delete_collection(collection_name=coll_name)
    # LanceDB.create_collection() does nothing, since we can't create a table
    # without a schema or data.
    vecdb.create_collection(collection_name=coll_name)

    # create random string of 10 arbitrary characters, not necessarily ascii
    import random

    # Generate a random string of 10 characters
    ingested_docs = [
        MyDocument(
            content=random.choice(["cow", "goat", "mouse"]),
            metadata=MyDocMeta(id=str(i), category=random.choice(["a", "b"])),
        )
        for i in range(20)
    ]

    vecdb.add_documents(ingested_docs)

    # test get ALL docs
    all_docs = vecdb.get_all_documents()
    ids = [doc.id() for doc in all_docs]
    assert len(set(ids)) == len(ids)
    assert len(all_docs) == len(ingested_docs)

    # test get docs by ids
    docs = vecdb.get_documents_by_ids(ids)
    assert len(docs) == len(ingested_docs)

    # test similarity search
    docs_and_scores = vecdb.similar_texts_with_scores("cow", k=1)
    assert len(docs_and_scores) == 1
    assert docs_and_scores[0][0].content == "cow"

    # test collections: create, list, clear
    if isinstance(vecdb, PineconeDB):
        # pinecone only allows lowercase alphanumeric with "-" characters
        coll_names = [f"test-junk-{i}" for i in range(3)]
    elif isinstance(vecdb, WeaviateDB):
        # Weaviate enforces capitalized collection names;
        # verifying adherence.

        coll_names = [f"Test_junk_{i}" for i in range(3)]
        for coll in coll_names:
            vecdb.create_collection(collection_name=coll)
        n_colls = len(
            [c for c in vecdb.list_collections(empty=True) if c.startswith("Test_junk")]
        )
        n_dels = vecdb.clear_all_collections(really=True, prefix="Test_junk")
    else:
        coll_names = [f"test_junk_{i}" for i in range(3)]
        for coll in coll_names:
            vecdb.create_collection(collection_name=coll)
        n_colls = len(
            [c for c in vecdb.list_collections(empty=True) if c.startswith("test_junk")]
        )
        n_dels = vecdb.clear_all_collections(really=True, prefix="test_junk")

    # LanceDB.create_collection() does nothing, since we can't create a table
    # without a schema or data.
    assert n_colls == n_dels == (0 if isinstance(vecdb, LanceDB) else len(coll_names))
    vecdb.set_collection(coll_name, replace=True)
    assert vecdb.config.collection_name == coll_name
    assert vecdb.get_all_documents() == []


@pytest.mark.parametrize(
    "vecdb",
    [
        "postgres",
        "lancedb",
        "chroma",
        "qdrant_cloud",
        "qdrant_local",
        pytest.param("pinecone_serverless", marks=pytest.mark.skip),
        "weaviate_docker",
    ],
    indirect=True,
)
def test_vector_stores_context_window(vecdb):
    """Test whether retrieving context-window around matches is working."""

    phrases = SimpleNamespace(
        CATS="Cats are quiet and clean.",
        DOGS="Dogs are noisy and messy.",
        GIRAFFES="Giraffes are tall and quiet.",
        ELEPHANTS="Elephants are big and noisy.",
        OWLS="Owls are quiet and nocturnal.",
        BATS="Bats are nocturnal and noisy.",
    )
    text = "\n\n".join(vars(phrases).values())
    doc = Document(content=text, metadata=DocMetaData(id="0"))
    cfg = ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_neighbor_ids=2,
        chunk_size=1,
        max_chunks=20,
        min_chunk_chars=3,
        discard_chunk_chars=1,
    )

    parser = Parser(cfg)
    splits = parser.split([doc])

    vecdb.create_collection(collection_name="testcw", replace=True)
    vecdb.add_documents(splits)

    # Test context window retrieval
    docs_scores = vecdb.similar_texts_with_scores("What are Giraffes like?", k=1)
    docs_scores = vecdb.add_context_window(docs_scores, neighbors=2)

    assert len(docs_scores) == 1
    giraffes, score = docs_scores[0]
    assert all(
        p in giraffes.content
        for p in [
            phrases.CATS,
            phrases.DOGS,
            phrases.GIRAFFES,
            phrases.ELEPHANTS,
            phrases.OWLS,
        ]
    )
    # check they are in the right sequence
    indices = [
        giraffes.content.index(p)
        for p in ["Cats", "Dogs", "Giraffes", "Elephants", "Owls"]
    ]

    assert indices == sorted(indices)


@pytest.mark.parametrize(
    "vecdb",
    [
        "qdrant_local",
        "qdrant_cloud",
        pytest.param("pinecone_serverless", marks=pytest.mark.skip),
        "chroma",
        "lancedb",
        # "postgres",
        "weaviate_docker",
    ],
    indirect=True,
)
def test_doc_chat_batch_with_vecdb_cloning(vecdb, test_settings):
    """Ensure DocChatAgent batching works with cloned vector stores."""

    cfg = DocChatAgentConfig(
        name=f"DocChatBatch-{vecdb.__class__.__name__}",
        vecdb=vecdb.config.model_copy(deep=True),
        retrieve_only=True,
        use_fuzzy_match=False,
        use_bm25_search=False,
        n_query_rephrases=0,
        hypothetical_answer=False,
    )

    try:
        agent = DocChatAgent(cfg)
    except LangroidImportError as exc:
        pytest.skip(
            f"Optional dependency missing for {vecdb.__class__.__name__}: {exc}"
        )
    except OperationalError as exc:
        pytest.skip(f"Database unavailable for {vecdb.__class__.__name__}: {exc}")
    except Exception as exc:
        pytest.skip(f"Skipping {vecdb.__class__.__name__} due to init failure: {exc}")

    agent.llm = None  # retrieval-only, avoid external LLM calls
    agent.vecdb.add_documents(stored_docs)
    agent.setup_documents()
    task = Task(agent, interactive=False, single_round=True)

    queries = ["hello", "hi there", "people living in Canada"]

    results = run_batch_tasks(
        task,
        queries,
        sequential=False,
        turns=1,
    )

    for query, result in zip(queries, results):
        assert result is not None
        assert hasattr(result, "content")
        assert query.lower() in result.content.lower()


@pytest.mark.parametrize(
    "vecdb",
    [
        "postgres",
        "chroma",
        "lancedb",
        "qdrant_cloud",
        "qdrant_local",
        pytest.param("pinecone_serverless", marks=pytest.mark.skip),
        "weaviate_docker",
    ],
    indirect=True,
)
def test_vector_stores_overlapping_matches(vecdb):
    """Test that overlapping windows are handled correctly."""

    # The windows around the first two giraffe matches should overlap.
    # The third giraffe match should be in a separate window.
    phrases = SimpleNamespace(
        CATS="Cats are quiet and clean.",
        DOGS="Dogs are noisy and messy.",
        GIRAFFES="Giraffes are tall and quiet.",
        ELEPHANTS="Elephants are big and noisy.",
        OWLS="Owls are quiet and nocturnal.",
        GIRAFFES1="Giraffes eat a lot of leaves.",
        COWS="Cows are quiet and gentle.",
        BULLS="Bulls are noisy and aggressive.",
        TIGERS="Tigers are big and noisy.",
        LIONS="Lions are nocturnal and noisy.",
        CHICKENS="Chickens are quiet and gentle.",
        ROOSTERS="Roosters are noisy and aggressive.",
        GIRAFFES3="Giraffes are really strange animals.",
        MICE="Mice are puny and gentle.",
        RATS="Rats are noisy and destructive.",
    )
    text = "\n\n".join(vars(phrases).values())
    doc = Document(content=text, metadata=DocMetaData(id="0"))

    cfg = ParsingConfig(
        splitter=Splitter.SIMPLE,
        n_neighbor_ids=2,
        chunk_size=1,
        max_chunks=20,
        min_chunk_chars=3,
        discard_chunk_chars=1,
    )

    parser = Parser(cfg)
    splits = parser.split([doc])

    vecdb.create_collection(collection_name="testcw", replace=True)
    vecdb.add_documents(splits)

    # Test context window retrieval
    docs_scores = vecdb.similar_texts_with_scores("What are Giraffes like?", k=3)
    # We expect to retrieve a window of -2, +2 around each of the three Giraffe matches.
    # The first two windows will overlap, so they form a connected component,
    # and we topological-sort and order the chunks in these windows, resulting in a
    # single window. The third Giraffe-match context window will not overlap with
    # the other two, so we will have a total of 2 final docs_scores components.
    docs_scores = vecdb.add_context_window(docs_scores, neighbors=2)

    assert len(docs_scores) == 2
    # verify no overlap in d.metadata.window_ids for d in docs
    all_window_ids = [id for d, _ in docs_scores for id in d.metadata.window_ids]
    assert len(all_window_ids) == len(set(all_window_ids))

    # verify giraffe occurs in each /match
    assert all("Giraffes" in d.content for d, _ in docs_scores)

    # verify correct sequence of chunks in each match
    sentences = vars(phrases).values()
    for d, _ in docs_scores:
        content = d.content
        indices = [content.find(p) for p in sentences]
        indices = [i for i in indices if i >= 0]
        assert indices == sorted(indices)


def test_lance_metadata():
    """
    Test that adding documents with extra fields in metadata
    (that are absent in the metadata of LanceDBConfig.document_class)
    works as expected, i.e. the internal schemas and config.document_class
    are dynamically updated as expected.
    """

    ldb_dir = ".lancedb/data/test"
    rmdir(ldb_dir)
    DEFAULT_COLLECTION = "test-dummy"
    ACTUAL_COLLECTION = "test-metadata"
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name=DEFAULT_COLLECTION,
        storage_path=ldb_dir,
        embedding=embed_cfg,
        document_class=Document,
    )
    vecdb = LanceDB(ldb_cfg)
    vecdb.set_collection(collection_name=ACTUAL_COLLECTION, replace=True)
    doc = Document(
        content="xyz",
        metadata=DocMetaData(
            id="0",
            source="wiki",
            category="other",  # this is an extra field not defined in DocMetaData
        ),
    )
    # since we're adding a document whose metadata has an extra field,
    # the config.document_class is updated to reflect the new schema.
    # and the schema is updated to accommodate the extra field,
    vecdb.add_documents([doc])

    # re-init the vecdb like above
    vecdb = LanceDB(ldb_cfg)

    # set to the SAME collection, so we don't create a new one
    vecdb.set_collection(collection_name=ACTUAL_COLLECTION, replace=False)

    # adding a new doc to an existing collection, it has a structure
    # consistent with the previous doc added to this collection,
    # BUT NOT consistent with the DEFAULT_COLLECTION.
    # We want to check that this goes well.
    doc = Document(
        content="abc",
        metadata=DocMetaData(
            category="main",  # this is an extra field not defined in DocMetaData
            id="1",
            source="wiki",
        ),
    )
    vecdb.add_documents([doc])

    doc = Document(
        content="abc",
        metadata=DocMetaData(
            id="2",
            category="rumor",  # this is an extra field not defined in DocMetaData
            source="web",
        ),
    )
    vecdb.add_documents([doc])

    all_docs = vecdb.get_all_documents()
    assert len(all_docs) == 3


@pytest.mark.parametrize(
    "vecdb",
    [
        "postgres",
    ],
    indirect=True,
)
def test_postgres_where_clause(vecdb: PostgresDB):
    """Test the where clause in get_all_documents,get_similar_texts in PostgresDB"""
    vecdb.create_collection(
        collection_name="test_get_all_documents_where", replace=True
    )
    docs = [
        Document(
            content="xyz",
            metadata=DocMetaData(
                id=str(i),
                source="wiki" if i % 2 == 0 else "web",
                category="other" if i < 3 else "news",
            ),
        )
        for i in range(5)
    ]
    vecdb.add_documents(docs)

    all_docs = vecdb.get_all_documents(where=json.dumps({"category": "other"}))
    assert len(all_docs) == 3

    all_docs = vecdb.get_all_documents(where=json.dumps({"source": "web"}))
    assert len(all_docs) == 2

    all_docs = vecdb.get_all_documents(
        where=json.dumps({"category": "other", "source": "web"})
    )
    assert len(all_docs) == 1

    all_docs = vecdb.get_all_documents(where=json.dumps({"category": "news"}))
    assert len(all_docs) == 2

    all_docs = vecdb.get_all_documents(where=json.dumps({"source": "wiki"}))
    assert len(all_docs) == 3

    vecdb.delete_collection("test_get_all_documents_where")
