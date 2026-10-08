from typing import List

import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.code_parser import CodeParser, CodeParsingConfig
from langroid.parsing.parser import Parser, ParsingConfig

MAX_CHUNK_SIZE = 10


def test_code_parser():
    cfg = CodeParsingConfig(
        chunk_size=MAX_CHUNK_SIZE,
        extensions=["py", "sh"],
        token_encoding_model="text-embedding-3-small",
    )

    parser = CodeParser(cfg)

    codes = """
    py|
    from pydantic import BaseModel
    from typing import List
    
    class Item(BaseModel):
        name: str
        description: str
        price: float
        tags: List[str]
    +
    py|
    import requests
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    +
    sh|
    #!/bin/bash

    # Function to prompt for user confirmation
    confirm() {
      read -p "$1 (y/n): " choice
      case "$choice" in
        [Yy]* ) return 0;;
        [Nn]* ) return 1;;
        * ) echo "Please answer y (yes) or n (no)."; return 1;;
      esac
    }
    """.split(
        "+"
    )

    codes = [text.strip() for text in codes if text.strip() != ""]
    lang_codes = [text.split("|") for text in codes]

    docs = [
        Document(content=code, metadata=DocMetaData(language=lang))
        for lang, code in lang_codes
        if code.strip() != ""
    ]
    split_docs = parser.split(docs)
    toks = parser.num_tokens
    # verify all chunks are less than twice max chunk size
    assert max([toks(doc.content) for doc in split_docs]) <= 2 * MAX_CHUNK_SIZE
    joined_splits = "".join([doc.content for doc in split_docs])
    joined_docs = "".join([doc.content for doc in docs])
    assert joined_splits.strip() == joined_docs.strip()


@pytest.fixture
def code_documents() -> List[Document]:
    """Return two source documents that each produce multiple code chunks."""
    return [
        Document(
            content="\n".join(f"value_{i} = {i}" for i in range(10)),
            metadata=DocMetaData(
                source="first.py", language="py", attributes={"kind": "code"}
            ),
        ),
        Document(
            content="\n".join(f"echo value_{i}" for i in range(10)),
            metadata=DocMetaData(source="second.sh", language="sh"),
        ),
    ]


def test_code_parser_metadata(code_documents: List[Document]) -> None:
    """Keep chunk metadata independent while preserving source metadata."""
    original = [doc.model_dump() for doc in code_documents]
    parser = CodeParser(CodeParsingConfig(chunk_size=MAX_CHUNK_SIZE))
    chunks = parser.split(code_documents)

    assert len({id(chunk.metadata) for chunk in chunks}) == len(chunks)
    for doc in code_documents:
        group = [c for c in chunks if c.metadata.source == doc.metadata.source]
        assert len(group) > 1
        assert "".join(c.content for c in group).strip() == doc.content.strip()
        assert all(c.metadata is not doc.metadata for c in group)
        # everything describing the ORIGIN is inherited; what identifies the
        # chunk itself (id, is_chunk, window_ids) is deliberately its own
        inherited = doc.metadata.model_dump(exclude={"id", "is_chunk", "window_ids"})
        assert all(
            c.metadata.model_dump(exclude={"id", "is_chunk", "window_ids"}) == inherited
            for c in group
        )

    # Nested values retain the existing shallow-copy semantics.
    assert chunks[0].metadata.attributes is code_documents[0].metadata.attributes
    chunks[0].metadata.source = "changed"
    assert chunks[1].metadata.source == "first.py"
    assert [doc.model_dump() for doc in code_documents] == original


@pytest.mark.parametrize("n_neighbor_ids", [0, 1, 3])
def test_code_parser_window_ids(
    code_documents: List[Document], n_neighbor_ids: int
) -> None:
    """`split` alone assigns distinct ids and source-local neighbor windows.

    The caller does not have to call `add_window_ids`: every other splitter
    does it internally, and a user who did not know to call it got chunks
    that all shared one id and overwrote each other on upsert.
    """
    original = [doc.model_dump() for doc in code_documents]
    chunks = CodeParser(
        CodeParsingConfig(chunk_size=MAX_CHUNK_SIZE, n_neighbor_ids=n_neighbor_ids)
    ).split(code_documents)

    assert len({chunk.id() for chunk in chunks}) == len(chunks)
    assert all(chunk.metadata.is_chunk for chunk in chunks)
    for doc in code_documents:
        group = [c for c in chunks if c.metadata.source == doc.metadata.source]
        ids = [chunk.id() for chunk in group]
        assert len(ids) > 1
        assert doc.id() not in ids
        for i, chunk in enumerate(group):
            assert (
                chunk.metadata.window_ids
                == ids[max(0, i - n_neighbor_ids) : i + n_neighbor_ids + 1]
            )
    assert [doc.model_dump() for doc in code_documents] == original


@pytest.mark.parametrize("n_neighbor_ids", [0, 1, 3])
def test_code_parser_add_window_ids_again_is_a_no_op(
    code_documents: List[Document], n_neighbor_ids: int
) -> None:
    """A user who still calls `add_window_ids` keeps their windows.

    `add_window_ids` groups chunks by a shared `metadata.id`, so on
    already-windowed input it would see each chunk as its own single-chunk
    document and collapse every window to a singleton -- silently losing
    the neighbors that `split` just assigned.
    """
    chunks = CodeParser(
        CodeParsingConfig(chunk_size=MAX_CHUNK_SIZE, n_neighbor_ids=n_neighbor_ids)
    ).split(code_documents)
    before = [(c.id(), list(c.metadata.window_ids)) for c in chunks]
    assert all(len(w) > 0 for _, w in before)

    # a different n_neighbor_ids, to catch a re-window that silently "works"
    Parser(ParsingConfig(n_neighbor_ids=n_neighbor_ids + 2)).add_window_ids(chunks)

    assert [(c.id(), list(c.metadata.window_ids)) for c in chunks] == before


def test_add_window_ids_still_windows_unchunked_input() -> None:
    """The no-op guard must not stop the first, real windowing pass.

    Counterpart to the guard above: fresh chunks share one id and carry no
    window, and those must still get distinct ids and neighbors.
    """
    source = DocMetaData(source="a.py", language="py")
    chunks = [
        Document(content=f"value_{i} = {i}", metadata=source.model_copy())
        for i in range(4)
    ]
    assert len({c.id() for c in chunks}) == 1

    Parser(ParsingConfig(n_neighbor_ids=1)).add_window_ids(chunks)

    ids = [c.id() for c in chunks]
    assert len(set(ids)) == 4
    assert all(c.metadata.is_chunk for c in chunks)
    assert [list(c.metadata.window_ids) for c in chunks] == [
        ids[0:2],
        ids[0:3],
        ids[1:4],
        ids[2:4],
    ]


def test_code_parser_resplit_still_gets_distinct_ids(
    code_documents: List[Document],
) -> None:
    """Splitting already-split chunks again still gives each a distinct id.

    The chunks of one source share an id, and if they also inherited that
    source's `window_ids` they would look already-windowed -- so the no-op
    guard would skip them and they would go back to sharing one id, which
    is the very bug this issue is about.
    """
    cfg = CodeParsingConfig(chunk_size=MAX_CHUNK_SIZE)
    big = CodeParser(CodeParsingConfig(chunk_size=10_000)).split(code_documents)
    assert len(big) == len(code_documents)  # one chunk per source

    small = CodeParser(cfg).split(big)

    assert len(small) > len(big)
    assert len({chunk.id() for chunk in small}) == len(small)
    assert all(len(c.metadata.window_ids) > 0 for c in small)
    # no chunk kept its parent's window
    assert all(c.metadata.window_ids != big[0].metadata.window_ids for c in small)


def test_add_window_ids_mixed_batch_keeps_existing_windows() -> None:
    """Windowed and fresh chunks in one call: only the fresh ones change.

    The guard is per group, not per call. A single un-windowed chunk in the
    batch must not force a re-window of everything, which would collapse
    the already-assigned windows to singletons.
    """
    shared = DocMetaData(source="a.py", language="py")
    done = [
        Document(content=f"value_{i} = {i}", metadata=shared.model_copy())
        for i in range(3)
    ]
    Parser(ParsingConfig(n_neighbor_ids=1)).add_window_ids(done)
    ids_before = [c.id() for c in done]
    windows_before = [list(c.metadata.window_ids) for c in done]
    assert [len(w) for w in windows_before] == [2, 3, 2]

    fresh = Document(content="echo hi", metadata=DocMetaData(source="b.sh"))
    Parser(ParsingConfig(n_neighbor_ids=1)).add_window_ids(done + [fresh])

    assert [c.id() for c in done] == ids_before
    assert [list(c.metadata.window_ids) for c in done] == windows_before
    assert fresh.metadata.window_ids == [fresh.id()]
    assert fresh.metadata.is_chunk


def test_code_parser_single_chunk_resplit_gets_a_fresh_id(
    code_documents: List[Document],
) -> None:
    """A re-split that yields ONE chunk per source still re-ids it.

    This is the shape that reaches the no-op guard, which only fires on a
    group of size 1: if the chunk inherited its parent's `window_ids`,
    that window contains the parent's id -- which is also the chunk's
    inherited id -- so the guard would take it for already-windowed and
    leave it sharing the parent's id.
    """
    huge = CodeParsingConfig(chunk_size=10_000)
    first = CodeParser(huge).split(code_documents)
    assert len(first) == len(code_documents)  # one chunk per source
    assert all(len(c.metadata.window_ids) > 0 for c in first)

    second = CodeParser(huge).split(first)

    assert len(second) == len(first)  # still one chunk per source
    parent_ids = {c.id() for c in first}
    assert all(c.id() not in parent_ids for c in second)
    assert len({c.id() for c in second}) == len(second)
    # each chunk's window is its own, naming itself rather than its parent
    for chunk in second:
        assert chunk.metadata.window_ids == [chunk.id()]


def test_parser_single_chunk_resplit_gets_a_fresh_id() -> None:
    """Same shape through `Parser.split_simple`, which shares the guard.

    `Parser.split` filters already-chunked docs out before splitting, so
    this is reachable only by calling a `split_*` method directly -- but
    it is the same defect, and the splitters inherit the same guard.
    """
    parser = Parser(ParsingConfig(n_neighbor_ids=1, separators=["\n"]))
    doc = Document(content="one line", metadata=DocMetaData(source="a.txt"))
    first = parser.split_simple([doc])
    assert len(first) == 1
    assert first[0].metadata.window_ids == [first[0].id()]

    second = parser.split_simple(first)

    assert len(second) == 1
    assert second[0].id() != first[0].id()
    assert second[0].metadata.window_ids == [second[0].id()]


def test_add_window_ids_sets_is_chunk_even_when_it_skips() -> None:
    """The guard still marks a chunk, as this function always has.

    It returns early for an already-windowed chunk, but `is_chunk` is a
    claim about the document, not about the window, and every caller used
    to be able to rely on `add_window_ids` setting it.
    """
    metadata = DocMetaData(source="a.py", language="py")
    metadata.window_ids = [metadata.id]
    chunk = Document(content="value = 1", metadata=metadata)
    assert not chunk.metadata.is_chunk

    Parser(ParsingConfig(n_neighbor_ids=1)).add_window_ids([chunk])

    assert chunk.metadata.is_chunk
    # and its window was left alone
    assert chunk.metadata.window_ids == [chunk.id()]
