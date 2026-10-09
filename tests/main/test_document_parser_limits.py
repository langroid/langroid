"""Resource-exhaustion limits in the document-parsing path.

Two separate guards are covered:

- `DocumentParser.get_doc_chunks` must honor `ParsingConfig.max_chunks`. It
  previously tracked a chunk count but never compared it against the limit, so
  one page expanding to millions of tokens produced an unbounded number of
  chunks.
- `DocumentParser.__init__` must reject a ZIP-based document (DOCX/XLSX/PPTX)
  whose members expand far beyond the compressed archive. `url_max_size`
  bounds only compressed HTTP bytes, and local paths / raw bytes are not
  bounded at all.

These tests use synthetic inputs only: no network, no LLM, no optional parser
backends.
"""

import random
from io import BytesIO
from typing import Any, Generator, List, Tuple
from zipfile import ZIP_DEFLATED, ZIP_LZMA, ZipFile

import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig


class _FakePagesParser(DocumentParser):
    """DocumentParser over in-memory page strings, with no real file format."""

    def __init__(self, page_texts: List[str], config: ParsingConfig) -> None:
        self._page_texts = page_texts
        super().__init__(b"", config)

    def iterate_pages(self) -> Generator[Tuple[int, Any], None, None]:
        for i, text in enumerate(self._page_texts):
            yield i, text

    def get_document_from_page(self, page: Any) -> Document:
        return Document(content=str(page), metadata=DocMetaData(source=self.source))


def _parsing_config(**kwargs: Any) -> ParsingConfig:
    return ParsingConfig(chunk_size=10, overlap=2, **kwargs)


def _zip_archive(member_bytes: bytes, name: str = "word/document.xml") -> bytes:
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr(name, member_bytes)
    return buf.getvalue()


def test_tests_run_against_this_checkout() -> None:
    """Guard against importing a different, installed `langroid`.

    Without this, a shadowed import makes every assertion below a statement
    about some other tree, and the counter-verification silently passes.
    """
    import pathlib

    import langroid.parsing.document_parser as mod

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    assert (
        pathlib.Path(mod.__file__).resolve().is_relative_to(repo_root)
    ), f"imported langroid from {mod.__file__}, not from {repo_root}"


def test_get_doc_chunks_honors_max_chunks() -> None:
    """One huge page must not produce more chunks than `max_chunks`."""
    # ~20k whitespace-separated tokens on a single page; with chunk_size=10
    # and overlap=2 that is well over 2000 chunks if unbounded.
    page = " ".join(f"w{i}" for i in range(20_000))
    config = _parsing_config(max_chunks=25)
    chunks = _FakePagesParser([page], config).get_doc_chunks()
    assert len(chunks) == 25


def test_get_doc_chunks_max_chunks_across_pages() -> None:
    """The cap also applies when the chunks come from many pages."""
    pages = [" ".join(f"p{p}w{i}" for i in range(200)) for p in range(40)]
    config = _parsing_config(max_chunks=17)
    chunks = _FakePagesParser(pages, config).get_doc_chunks()
    assert len(chunks) == 17


def test_get_doc_chunks_under_the_cap_is_unchanged() -> None:
    """A document well under the cap keeps its full, untruncated chunking."""
    page = " ".join(f"w{i}" for i in range(300))
    unlimited = _FakePagesParser([page], _parsing_config()).get_doc_chunks()
    capped = _FakePagesParser(
        [page], _parsing_config(max_chunks=10_000)
    ).get_doc_chunks()
    assert 1 < len(unlimited) < 10_000
    assert [d.content for d in capped] == [d.content for d in unlimited]


def test_short_document_still_yields_one_chunk() -> None:
    """A document shorter than one chunk still yields exactly one chunk."""
    chunks = _FakePagesParser(["just a few words"], _parsing_config()).get_doc_chunks()
    assert len(chunks) == 1
    assert "just a few words" in chunks[0].content


def test_trailing_chunk_does_not_exceed_max_chunks() -> None:
    """The final partial chunk must not push the total one past the cap.

    The chunking loop stops as soon as the remainder is shorter than one
    chunk, so the trailing-chunk branch is reached without the loop ever
    observing the cap. With chunk_size=10, overlap=2 and max_chunks=1, eleven
    tokens produce one full chunk plus a three-token remainder.
    """
    page = "a b c d e f g h i j k"
    parser = _FakePagesParser([page], _parsing_config(max_chunks=1))
    # The boundary only exists at exactly 11 tokens: 12 or more and the inner
    # loop sees the cap itself, which is a different branch. Assert the
    # precondition so a tokenizer change cannot quietly defuse this test.
    assert len(parser.tokenizer.encode(page)) == 11
    assert len(parser.get_doc_chunks()) == 1


def test_zip_bomb_docx_is_rejected() -> None:
    """A DOCX whose XML expands past the budget is rejected before parsing."""
    expanded = b"<w:t>" + b"word " * 2_000_000 + b"</w:t>"
    archive = _zip_archive(expanded)
    config = ParsingConfig(zip_max_expanded_size=1_000_000)
    assert len(expanded) > 1_000_000 > len(archive)
    with pytest.raises(ValueError) as exc:
        DocumentParser.create(archive, config, doc_type="docx")
    assert "zip_max_expanded_size" in str(exc.value)


def test_zip_bomb_with_leading_junk_is_rejected() -> None:
    """Leading bytes before the ZIP header must not bypass the guard.

    `zipfile` finds the central directory from the END of the file, so an
    archive with arbitrary leading bytes still parses and still expands. A
    guard that sniffs a magic number at offset 0 would wave this through.
    """
    expanded = b"<w:t>" + b"word " * 2_000_000 + b"</w:t>"
    archive = b"JUNK" * 32 + _zip_archive(expanded)
    config = ParsingConfig(zip_max_expanded_size=1_000_000)
    # Sanity: the prefixed archive is still a readable ZIP with the big member.
    with ZipFile(BytesIO(archive)) as zf:
        assert len(zf.read("word/document.xml")) == len(expanded)
    with pytest.raises(ValueError) as exc:
        DocumentParser.create(archive, config, doc_type="docx")
    assert "zip_max_expanded_size" in str(exc.value)


def test_zip_metadata_cannot_understate_expanded_size() -> None:
    """The budget is measured from real decompressed bytes, not ZIP metadata."""
    expanded = b"<w:t>" + b"word " * 2_000_000 + b"</w:t>"
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("word/document.xml", expanded)
    raw = bytearray(buf.getvalue())
    # Lie about every recorded uncompressed size (local header + central dir).
    for info_offset in range(len(raw) - 4):
        if raw[info_offset : info_offset + 4] in (b"PK\x03\x04", b"PK\x01\x02"):
            field = info_offset + (22 if raw[info_offset + 3] == 0x04 else 24)
            raw[field : field + 4] = (1).to_bytes(4, "little")
    config = ParsingConfig(zip_max_expanded_size=1_000_000)
    # `zipfile` stops at the recorded size and then fails its CRC check, which
    # the guard converts into a refusal rather than letting it escape raw.
    with pytest.raises(ValueError) as exc:
        DocumentParser.create(bytes(raw), config, doc_type="docx")
    assert "cannot be bounded" in str(exc.value)


def test_lzma_member_is_rejected() -> None:
    """A member whose expansion cannot be bounded step by step is refused.

    `zipfile` bounds output per `read()` only for stored and deflated members;
    the LZMA and bzip2 decompressors accept no output bound, so one read can
    materialize a whole member internally -- the guard would allocate the bomb
    before its own counter noticed.
    """
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_LZMA) as archive:
        archive.writestr("word/document.xml", b"A" * (16 * 1024 * 1024))
    assert len(buf.getvalue()) < 10_000  # a tiny archive, 16 MiB expanded
    with pytest.raises(ValueError) as exc:
        DocumentParser.create(buf.getvalue(), ParsingConfig(), doc_type="docx")
    assert "cannot be bounded" in str(exc.value)


def test_directory_named_member_is_still_measured() -> None:
    """A payload in an entry whose name ends in "/" must not escape the budget.

    `ZipInfo.is_dir()` only looks at the name, but a consumer that opens the
    part its manifest names will read the entry's data anyway, so a guard that
    skipped such entries would leave a hole exactly the size of the payload.
    """
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<w:t>small</w:t>")
        archive.writestr("payload/", b"word " * 400_000)
    config = ParsingConfig(zip_max_expanded_size=100_000)
    with pytest.raises(ValueError) as exc:
        DocumentParser.create(buf.getvalue(), config, doc_type="docx")
    assert "zip_max_expanded_size" in str(exc.value)


def test_ordinary_zip_document_is_accepted() -> None:
    """A realistic small DOCX passes the guard (construction must succeed)."""
    archive = _zip_archive(b"<w:t>" + b"hello world " * 50 + b"</w:t>")
    parser = DocumentParser.create(archive, ParsingConfig(), doc_type="docx")
    assert parser.doc_bytes.tell() == 0  # rewound, ready for the real parser


def test_repetitive_but_legitimate_docx_is_accepted() -> None:
    """A real, highly compressible DOCX must not be mistaken for a bomb.

    Thousands of similar paragraphs or table rows are ordinary OOXML and
    compress by several hundred to one, which is why only an absolute byte
    budget is enforced and no compression-ratio limit.
    """
    body = b"<w:p><w:r><w:t>Pending</w:t></w:r></w:p>" * 30_000
    archive = _zip_archive(b"<w:document><w:body>" + body + b"</w:body></w:document>")
    assert len(body) / len(archive) > 100.0  # a ratio limit would reject this
    assert len(body) < ParsingConfig().zip_max_expanded_size  # under the budget
    parser = DocumentParser.create(archive, ParsingConfig(), doc_type="docx")
    assert parser.doc_bytes.tell() == 0


def test_zip_guard_can_be_disabled() -> None:
    """`zip_max_expanded_size=0` turns the guard off."""
    archive = _zip_archive(b"<w:t>" + b"word " * 2_000_000 + b"</w:t>")
    config = ParsingConfig(zip_max_expanded_size=0)
    parser = DocumentParser.create(archive, config, doc_type="docx")
    assert parser.doc_bytes.tell() == 0


def test_non_zip_bytes_pass_the_guard_untouched() -> None:
    """Non-ZIP input is left to the real parser to reject."""
    # Imported here, not at module scope: a module-scope import of a private
    # helper turns the whole file into a collection error on a tree without the
    # fix, which would make the counter-verification vacuous.
    from langroid.parsing.document_parser import _enforce_zip_expansion_limit

    pdf_bytes = b"%PDF-1.4\n" + random.Random(1).randbytes(200_000)
    buf = BytesIO(pdf_bytes)
    _enforce_zip_expansion_limit(buf, ParsingConfig(zip_max_expanded_size=1000))
    assert buf.read() == pdf_bytes
