"""Chroma ID retrieval against a local store with explicit embeddings."""

from pathlib import Path
from typing import Iterator, List

import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.vector_store.chromadb import ChromaDB, ChromaDBConfig


@pytest.fixture
def chroma_store(tmp_path: Path) -> Iterator[ChromaDB]:
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection("id-retrieval", embedding_function=None)
    documents = [
        Document(
            content=content,
            metadata=DocMetaData(
                id=doc_id,
                source=f"source-{doc_id}",
                is_chunk=True,
                window_ids=["a", "b", "c"],
            ),
        )
        for doc_id, content in zip(["a", "b", "c"], ["Alpha ", "Beta ", "Gamma"])
    ]
    metadatas = [doc.metadata.dict_bool_int() for doc in documents]
    for metadata in metadatas:
        metadata["window_ids"] = ",".join(metadata["window_ids"])
    collection.add(
        ids=[doc.id() for doc in documents],
        documents=[doc.content for doc in documents],
        metadatas=metadatas,
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    # Attach the real collection without constructing an embedding provider.
    store = object.__new__(ChromaDB)
    store.config = ChromaDBConfig(collection_name="id-retrieval")
    store.client = client
    store.collection = collection
    yield store
    client.delete_collection("id-retrieval")


@pytest.mark.parametrize(
    "ids, expected",
    [
        ([], []),
        (["missing"], []),
        (["missing", "also-missing"], []),
        (["c", "a"], ["c", "a"]),
        (["c", "a", "c"], ["c", "a", "c"]),
        (["missing", "c", "a"], ["c", "a"]),
        (["c", "missing", "a"], ["c", "a"]),
        (["c", "a", "missing"], ["c", "a"]),
        (["c", "missing", "a", "c", "missing"], ["c", "a", "c"]),
    ],
)
def test_chroma_get_documents_by_ids(
    chroma_store: ChromaDB, ids: List[str], expected: List[str]
) -> None:
    """Missing IDs are omitted without losing order or repeated matches."""
    documents = chroma_store.get_documents_by_ids(ids)
    assert [doc.id() for doc in documents] == expected
    contents = {"a": "Alpha ", "b": "Beta ", "c": "Gamma"}
    assert [doc.content for doc in documents] == [contents[id] for id in expected]
    assert [doc.metadata.source for doc in documents] == [
        f"source-{id}" for id in expected
    ]
    assert all(doc.metadata.window_ids == ["a", "b", "c"] for doc in documents)


def test_chroma_context_window_with_deleted_neighbor(chroma_store: ChromaDB) -> None:
    """An absent neighboring chunk does not discard the surviving context."""
    match = chroma_store.get_documents_by_ids(["a"])[0]
    chroma_store.collection.delete(ids=["b"])
    result = chroma_store.add_context_window([(match, 0.8)], neighbors=2)
    assert len(result) == 1
    document, score = result[0]
    assert document.content == "Alpha Gamma"
    assert document.metadata.source == "source-a"
    assert score == 0.8
