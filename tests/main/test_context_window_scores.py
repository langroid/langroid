"""Context-window scores with a local, in-memory document store."""

from typing import List, Optional, Sequence, Tuple

import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import add_window_ids
from langroid.vector_store.base import VectorStore


class MemoryVectorStore(VectorStore):
    """Exercise the shared windowing implementation without an external DB."""

    def __init__(self, documents: Sequence[Document]):
        self.documents = {doc.id(): doc for doc in documents}

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        return [self.documents[id_] for id_ in ids if id_ in self.documents]

    def get_all_documents(self, where: str = "") -> List[Document]:
        return list(self.documents.values())

    def add_documents(self, documents: Sequence[Document]) -> None:
        self.documents.update({doc.id(): doc for doc in documents})

    def similar_texts_with_scores(
        self, text: str, k: int = 1, where: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    def clear_empty_collections(self) -> int:
        raise NotImplementedError

    def clear_all_collections(self, really: bool = False, prefix: str = "") -> int:
        raise NotImplementedError

    def list_collections(self, empty: bool = False) -> List[str]:
        raise NotImplementedError

    def create_collection(self, collection_name: str, replace: bool = False) -> None:
        raise NotImplementedError

    def delete_collection(self, collection_name: str) -> None:
        raise NotImplementedError


def make_chunks(stored_neighbors: int) -> List[Document]:
    """Make chunks with the same neighbor metadata produced during ingestion."""
    chunks = [
        Document(
            content=f"Paragraph {i}.\n",
            metadata=DocMetaData(id="source-document"),
        )
        for i in range(9)
    ]
    add_window_ids(chunks, n_neighbor_ids=stored_neighbors)
    return chunks


@pytest.mark.parametrize("reverse_matches", [False, True])
@pytest.mark.parametrize("stored_neighbors", [1, 5])
def test_disjoint_windows_keep_their_own_scores(
    stored_neighbors: int, reverse_matches: bool
) -> None:
    """Unused stored neighbors must not leak a score to another output window."""
    chunks = make_chunks(stored_neighbors)
    store = MemoryVectorStore(chunks)
    matches = [(chunks[2], 0.9), (chunks[6], 0.2)]
    if reverse_matches:
        matches.reverse()

    results = store.add_context_window(matches, neighbors=1)

    assert len(results) == 2
    assert {doc.content: score for doc, score in results} == {
        "".join(doc.content for doc in chunks[1:4]): 0.9,
        "".join(doc.content for doc in chunks[5:8]): 0.2,
    }
    assert not set(results[0][0].metadata.window_ids).intersection(
        results[1][0].metadata.window_ids
    )


@pytest.mark.parametrize("reverse_matches", [False, True])
def test_overlapping_windows_keep_highest_match_score(
    reverse_matches: bool,
) -> None:
    """Windows that actually overlap still combine their match scores."""
    chunks = make_chunks(stored_neighbors=5)
    store = MemoryVectorStore(chunks)
    matches = [(chunks[2], 0.9), (chunks[4], 0.2)]
    if reverse_matches:
        matches.reverse()

    results = store.add_context_window(matches, neighbors=1)

    assert len(results) == 1
    document, score = results[0]
    assert document.content == "".join(doc.content for doc in chunks[1:6])
    assert score == 0.9


def test_zero_neighbors_returns_original_matches() -> None:
    chunks = make_chunks(stored_neighbors=5)
    store = MemoryVectorStore(chunks)
    matches = [(chunks[2], 0.9), (chunks[6], 0.2)]

    assert store.add_context_window(matches, neighbors=0) is matches


def test_no_stored_neighbors_keeps_original_scores() -> None:
    chunks = make_chunks(stored_neighbors=0)
    store = MemoryVectorStore(chunks)
    matches = [(chunks[2], 0.9), (chunks[6], 0.2)]

    results = store.add_context_window(matches, neighbors=1)

    assert [(doc.content, score) for doc, score in results] == [
        (doc.content, score) for doc, score in matches
    ]
