from typing import Sequence

import pytest

from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
    _append_metadata_source,
)
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import DocMetaData, Document


class _RecordingVectorStore:
    """Record documents passed through the ingestion path."""

    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add_documents(self, documents: Sequence[Document]) -> None:
        self.documents.extend(documents)


class _MetadataTestAgent(DocChatAgent):
    """Avoid unrelated lexical-index setup in the ingestion regression."""

    def setup_documents(
        self,
        docs: list[Document] = [],
        filter: str | None = None,
    ) -> None:
        pass


@pytest.mark.parametrize(
    ("original", "new", "expected"),
    [
        ("source-a", "source-a", "source-a"),
        ("  source-a  ", " source-a ", "source-a"),
        ("   ", " source-a ", "source-a"),
        (" source-a ", "\t", "source-a"),
        ("", "source-a", "source-a"),
        ("source-a", "", "source-a"),
        ("", "", ""),
        ("source-a", "source-b", "source-a; source-b"),
        ("Source-A", "source-a", "Source-A; source-a"),
        (" source-a ", " source-b ", "source-a; source-b"),
        ("https://host/a;b", "https://host/a;b", "https://host/a;b"),
        (
            "https://host/a;b; source-c",
            "https://host/a;b",
            "https://host/a;b; source-c",
        ),
        ("source-a; source-b", "source-b", "source-a; source-b"),
        ("source-a;  source-b ", "source-b", "source-a;  source-b"),
        (None, "source-a", "source-a"),
        ("source-a", None, "source-a"),
        ({"unexpected": "shape"}, "source-a", "source-a"),
        ("source-a", ["unexpected", "shape"], "source-a"),
    ],
)
def test_append_metadata_source(
    original: object,
    new: object,
    expected: str,
) -> None:
    assert _append_metadata_source(original, new) == expected


def test_ingest_docs_handles_invalid_source_shapes() -> None:
    """Ingestion keeps whichever source value is a valid string."""
    agent = _MetadataTestAgent(DocChatAgentConfig(llm=MockLMConfig(), vecdb=None))
    store = _RecordingVectorStore()
    agent.vecdb = store  # type: ignore[assignment]
    docs = [
        Document(content="one", metadata=DocMetaData(source="original")),
        Document(content="two", metadata=DocMetaData(source="unused")),
    ]
    docs[1].metadata.source = {"unexpected": "shape"}  # type: ignore[assignment]

    agent.ingest_docs(
        docs,
        split=False,
        metadata=[
            {"source": None},
            {"source": "replacement"},
        ],
    )

    assert [document.metadata.source for document in store.documents] == [
        "original",
        "replacement",
    ]
