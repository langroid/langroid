"""
Tests for Markdown and HTML support in DocumentParser / DocumentType.

Covers:
- DocumentType enum has MD and HTML values
- _document_type() dispatches on extension (case-insensitive) for all
  recognised extensions, and falls back to the plain-text heuristic (or
  ValueError) for unrecognised ones
- DocumentType.MD correctly precedes is_plain_text() heuristic
- chunks_from_path_or_bytes() returns non-empty chunks for both formats
- Markdown is returned as raw Markdown (syntax and embedded HTML-like
  text preserved); YAML front-matter is stripped
- HTML tags are stripped; adjacent elements are newline-separated
- MIME-based bytes dispatch (text/html, text/markdown, text/x-markdown,
  text/plain) is covered deterministically via a fake `magic` module
- .md / .html / .htm URLs are fetched over HTTP(S) and parsed
- DocumentParser.create() raises a clear ValueError for MD/HTML
  (they are handled via chunks_from_path_or_bytes, not via a subclass)
- Plain-text bytes ingestion falls back to the UTF-8 heuristic (and never
  errors out) when python-magic / native libmagic is missing or broken
"""

import functools
import http.server
import os
import pathlib
import shutil
import sys
import threading
import time
import types
from typing import Any, Iterator

import pytest
import requests

from langroid.parsing.document_parser import (
    DocumentParser,
    DocumentType,
    _strip_yaml_frontmatter,
)
from langroid.parsing.parser import Parser, ParsingConfig, Splitter

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_MD_PATH = os.path.join(_DATA_DIR, "sample.md")
_HTML_PATH = os.path.join(_DATA_DIR, "sample.html")


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def test_document_type_enum_has_md_and_html() -> None:
    assert DocumentType.MD == "md"
    assert DocumentType.HTML == "html"


# ---------------------------------------------------------------------------
# _document_type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("report.md", DocumentType.MD),
        ("notes.MD", DocumentType.MD),
        ("page.html", DocumentType.HTML),
        ("page.htm", DocumentType.HTML),
        ("PAGE.HTML", DocumentType.HTML),
        ("page.HtM", DocumentType.HTML),
        # Legacy extensions must dispatch case-insensitively, BEFORE the
        # plain-text heuristic (none of these files exist on disk).
        ("paper.pdf", DocumentType.PDF),
        ("Paper.PDF", DocumentType.PDF),
        ("report.docx", DocumentType.DOCX),
        ("Report.DocX", DocumentType.DOCX),
        ("legacy.doc", DocumentType.DOC),
        ("LEGACY.DOC", DocumentType.DOC),
        ("sheet.xlsx", DocumentType.XLSX),
        ("Sheet.XLSX", DocumentType.XLSX),
        ("old.xls", DocumentType.XLS),
        ("Old.XlS", DocumentType.XLS),
        ("slides.pptx", DocumentType.PPTX),
        ("Slides.PpTx", DocumentType.PPTX),
    ],
)
def test_document_type_extension_detection(
    filename: str, expected: DocumentType
) -> None:
    # Use _document_type directly; pass a fake path with the right extension.
    # The file doesn't need to exist — we only care about extension dispatch.
    result = DocumentParser._document_type(filename)
    assert result == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com/readme.md?version=1", DocumentType.MD),
        ("https://example.com/README.md#intro", DocumentType.MD),
        ("https://example.com/page.html?raw=1", DocumentType.HTML),
        ("https://example.com/page.HTML#section", DocumentType.HTML),
        ("https://example.com/page.htm?raw=1#section", DocumentType.HTML),
    ],
)
def test_document_type_url_extension_detection_ignores_query_and_fragment(
    url: str, expected: DocumentType
) -> None:
    result = DocumentParser._document_type(url)
    assert result == expected


def test_unrecognized_extension_text_file_falls_back_to_txt(
    tmp_path: pathlib.Path,
) -> None:
    """Unknown extension + valid UTF-8 content → plain-text heuristic → TXT."""
    path = tmp_path / "notes.customext"
    path.write_text("Just ordinary readable text.\nSecond line of text.")
    assert DocumentParser._document_type(str(path)) == DocumentType.TXT


def test_unrecognized_extension_binary_file_raises_value_error(
    tmp_path: pathlib.Path,
) -> None:
    """Unknown extension + non-text content → ValueError."""
    path = tmp_path / "blob.customext"
    path.write_bytes(b"\x00\xff\xfe\x00\x01\x02binary\x80\x81\x00")
    with pytest.raises(ValueError, match="Unsupported document type"):
        DocumentParser._document_type(str(path))


def test_md_extension_not_classified_as_txt() -> None:
    """A .md file must not fall back to TXT via the is_plain_text heuristic."""
    result = DocumentParser._document_type(_MD_PATH)
    assert result == DocumentType.MD
    assert result != DocumentType.TXT


def test_html_extension_not_classified_as_txt() -> None:
    result = DocumentParser._document_type(_HTML_PATH)
    assert result == DocumentType.HTML
    assert result != DocumentType.TXT


# ---------------------------------------------------------------------------
# chunks_from_path_or_bytes — Markdown
# ---------------------------------------------------------------------------


def _assert_raw_markdown_preserved(full_text: str) -> None:
    """Assert content is the RAW Markdown source, not rendered/tag-stripped.

    These would all fail if the implementation rendered the Markdown or
    passed it through BeautifulSoup's get_text() (which drops/alters
    tag-like text such as ``<placeholder>``).
    """
    assert "# Heading One" in full_text
    assert "**Markdown**" in full_text
    assert "`code`" in full_text
    assert "[link](https://example.com)" in full_text
    assert "<placeholder>" in full_text


def test_md_chunks_from_path() -> None:
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(_MD_PATH, parser)

    assert len(chunks) > 0
    full_text = "\n".join(c.content for c in chunks)

    # Front-matter should be stripped
    assert "title:" not in full_text
    assert "author:" not in full_text

    # Body content must survive
    assert "Heading One" in full_text
    assert "Item A" in full_text
    assert "blockquote" in full_text or "blockquote paragraph" in full_text

    # Contract: MD returns the raw Markdown source (minus front-matter).
    _assert_raw_markdown_preserved(full_text)


def test_md_chunks_from_bytes() -> None:
    with open(_MD_PATH, "rb") as f:
        raw = f.read()
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="md")

    assert len(chunks) > 0
    full_text = "\n".join(c.content for c in chunks)
    assert "Heading One" in full_text

    # Contract: MD returns the raw Markdown source (minus front-matter).
    _assert_raw_markdown_preserved(full_text)


def test_md_frontmatter_stripped() -> None:
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(_MD_PATH, parser)
    full_text = " ".join(c.content for c in chunks)

    # YAML front-matter lines must not appear in output
    assert "---" not in full_text
    assert "title: Test Document" not in full_text


# ---------------------------------------------------------------------------
# chunks_from_path_or_bytes — HTML
# ---------------------------------------------------------------------------


def test_html_chunks_from_path() -> None:
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(_HTML_PATH, parser)

    assert len(chunks) > 0
    full_text = " ".join(c.content for c in chunks)

    # HTML tags must be stripped
    assert "<h1>" not in full_text
    assert "<p>" not in full_text
    assert "<li>" not in full_text

    # Readable text must survive
    assert "Heading One" in full_text
    assert "Item A" in full_text


def test_html_chunks_from_bytes() -> None:
    with open(_HTML_PATH, "rb") as f:
        raw = f.read()
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="html")

    assert len(chunks) > 0
    full_text = " ".join(c.content for c in chunks)
    assert "Heading One" in full_text
    assert "<h1>" not in full_text


def test_windows1252_html_chunks_from_bytes() -> None:
    html = (
        '<html><head><meta charset="windows-1252"></head>'
        "<body>Crème brûlée — déjà vu</body></html>"
    ).encode("windows-1252")

    chunks = DocumentParser.chunks_from_path_or_bytes(
        html, Parser(ParsingConfig()), doc_type="html"
    )

    assert chunks
    assert "Crème brûlée — déjà vu" in "\n".join(chunk.content for chunk in chunks)


def test_windows1252_html_chunks_from_local_file(
    tmp_path: pathlib.Path,
) -> None:
    html = (
        '<html><head><meta charset="windows-1252"></head>'
        "<body>Crème brûlée — déjà vu</body></html>"
    ).encode("windows-1252")
    path = tmp_path / "windows1252.html"
    path.write_bytes(html)

    chunks = DocumentParser.chunks_from_path_or_bytes(
        str(path), Parser(ParsingConfig())
    )

    assert chunks
    assert "Crème brûlée — déjà vu" in "\n".join(chunk.content for chunk in chunks)


def test_html_adjacent_elements_newline_separated_from_path() -> None:
    """Adjacent HTML elements must be newline-separated, not concatenated.

    sample.html contains ``<li>Item A</li><li>Item B</li>`` with no
    whitespace between the elements: a regression from
    ``get_text(separator="\\n")`` back to ``get_text()`` would concatenate
    them into ``Item AItem B``.
    """
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(_HTML_PATH, parser)
    full_text = "\n".join(c.content for c in chunks)
    assert "Item AItem B" not in full_text
    assert "Item A\nItem B" in full_text


def test_html_adjacent_elements_newline_separated_from_bytes() -> None:
    """Same newline-separation contract for the explicit HTML-bytes path."""
    html = b"<html><body><ul><li>A1</li><li>B2</li></ul><p>C3</p></body></html>"
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(html, parser, doc_type="html")
    full_text = "\n".join(c.content for c in chunks)
    assert "A1B2" not in full_text
    assert "A1\nB2\nC3" in full_text


def test_html_bytes_auto_detected_without_doc_type() -> None:
    """HTML bytes must be detected as HTML (not TXT) even without doc_type.

    Previously is_plain_text() short-circuited to TXT for any valid UTF-8
    bytes before MIME detection could run, so HTML bytes were routed through
    the BeautifulSoup/TXT path accidentally (same result here, but with the
    wrong DocumentType).  This test verifies the MIME path is reached.

    Requires libmagic (python-magic); skipped if the native library is absent.
    """
    magic = pytest.importorskip(
        "magic",
        reason="libmagic not available; skipping MIME-based bytes detection test",
    )
    # Verify that magic itself is functional (not just importable).
    try:
        magic.from_buffer(b"test", mime=True)
    except Exception:
        pytest.skip("libmagic not functional on this system")

    with open(_HTML_PATH, "rb") as f:
        raw = f.read()
    detected = DocumentParser._document_type(raw)
    assert detected == DocumentType.HTML


def _make_fake_magic(mime_type: str) -> types.ModuleType:
    """Build a fake ``magic`` module whose from_buffer returns `mime_type`."""
    fake = types.ModuleType("magic")

    def _from_buffer(*args: object, **kwargs: object) -> str:
        return mime_type

    fake.from_buffer = _from_buffer  # type: ignore[attr-defined]
    return fake


@pytest.mark.parametrize(
    "mime_type, expected",
    [
        ("text/html", DocumentType.HTML),
        ("text/markdown", DocumentType.MD),
        ("text/x-markdown", DocumentType.MD),
        ("text/plain", DocumentType.TXT),
    ],
)
def test_bytes_mime_type_dispatch(
    monkeypatch: pytest.MonkeyPatch, mime_type: str, expected: DocumentType
) -> None:
    """Bytes dispatch per detected MIME type, independent of libmagic.

    Deterministic version of the auto-detection tests: a fake ``magic``
    module pins the MIME type, so this runs on systems without libmagic.
    """
    monkeypatch.setitem(sys.modules, "magic", _make_fake_magic(mime_type))
    raw = b"---\ntitle: Front Matter\n---\n\n# Heading\n\nBody text."
    assert DocumentParser._document_type(raw) == expected


def test_text_plain_md_bytes_ingest_as_txt_and_keep_frontmatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown bytes MIME-detected as text/plain (no doc_type) become TXT.

    Documented limitation: libmagic reports most Markdown content as
    text/plain, so without ``doc_type="md"`` the bytes take the TXT path
    and YAML front-matter is NOT stripped by the document parser.

    Uses the TOKENS splitter: the default MARKDOWN splitter's chunker
    strips front-matter downstream, which would mask what the TXT path
    itself does.
    """
    monkeypatch.setitem(sys.modules, "magic", _make_fake_magic("text/plain"))
    with open(_MD_PATH, "rb") as f:
        raw = f.read()
    assert DocumentParser._document_type(raw) == DocumentType.TXT

    parser = Parser(ParsingConfig(splitter=Splitter.TOKENS))
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser)
    assert len(chunks) > 0
    full_text = "\n".join(c.content for c in chunks)
    # Front-matter is retained on the TXT path (doc_type="md" not passed).
    assert "title: Test Document" in full_text
    assert "Heading One" in full_text


def test_md_bytes_with_doc_type_strips_frontmatter() -> None:
    """Markdown bytes passed with doc_type='md' must strip YAML front-matter.

    Uses the TOKENS splitter so the stripping is attributable to the MD
    branch of chunks_from_path_or_bytes: the default MARKDOWN splitter's
    chunker also strips front-matter, which would mask a regression.
    """
    with open(_MD_PATH, "rb") as f:
        raw = f.read()
    parser = Parser(ParsingConfig(splitter=Splitter.TOKENS))
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="md")
    full_text = " ".join(c.content for c in chunks)
    assert "title:" not in full_text
    assert "Heading One" in full_text
    # Raw Markdown (including HTML-like tokens) survives on the MD path.
    assert "<placeholder>" in full_text


# ---------------------------------------------------------------------------
# _strip_yaml_frontmatter — unit tests
# ---------------------------------------------------------------------------


def test_strip_yaml_frontmatter_strips_valid_block() -> None:
    doc = "---\ntitle: My Doc\nauthor: Alice\n---\n\n# Heading\n\nBody text."
    result = _strip_yaml_frontmatter(doc)
    assert "title:" not in result
    assert "author:" not in result
    assert "Heading" in result
    assert "Body text" in result


def test_strip_yaml_frontmatter_strips_valid_block_at_eof() -> None:
    """A valid front-matter block may end at EOF."""
    result = _strip_yaml_frontmatter("---\ntitle: Secret\n---")
    assert result == ""


def test_strip_yaml_frontmatter_preserves_thematic_break() -> None:
    """A leading --- thematic break (no key: value inside) must be kept."""
    doc = "---\n\nSome intro text.\n\n---\n\n# Section"
    result = _strip_yaml_frontmatter(doc)
    assert "Some intro text" in result


def test_strip_yaml_frontmatter_preserves_content_between_dashes() -> None:
    """A leading --- block whose content has no YAML keys must be kept."""
    doc = "---\nThis is just a horizontal rule section.\n---\n\nReal content here."
    result = _strip_yaml_frontmatter(doc)
    assert "horizontal rule" in result
    assert "Real content" in result


def test_strip_yaml_frontmatter_no_frontmatter() -> None:
    """A document with no --- block is returned unchanged (stripped)."""
    doc = "# Title\n\nJust a normal document."
    result = _strip_yaml_frontmatter(doc)
    assert result == doc.strip()


def test_strip_yaml_frontmatter_partial_yaml_keys() -> None:
    """Even a single key: value line qualifies the block as YAML front-matter."""
    doc = "---\ntitle: Single Key\n---\n\nContent."
    result = _strip_yaml_frontmatter(doc)
    assert "title:" not in result
    assert "Content" in result


def test_strip_yaml_frontmatter_url_not_treated_as_yaml() -> None:
    """A block with only a URL (colon, but not a YAML key) must be kept."""
    doc = "---\nhttps://example.com\n---\n\nReal content."
    result = _strip_yaml_frontmatter(doc)
    assert "example.com" in result
    assert "Real content" in result


def test_strip_yaml_frontmatter_preserves_leading_indentation() -> None:
    """Leading spaces after front-matter (e.g. indented code) must survive."""
    doc = "---\ntitle: Doc\n---\n\n    indented code block\n\nNormal paragraph."
    result = _strip_yaml_frontmatter(doc)
    # The indented code block's spaces must survive
    assert "    indented code block" in result
    assert "Normal paragraph" in result


def test_strip_yaml_frontmatter_prose_with_colon() -> None:
    """Only bare-identifier `key:` lines qualify a block as front-matter.

    Known limitation (inherent Markdown/YAML ambiguity): a leading divider
    block like ``---\\nNote: important\\n---`` IS stripped, since ``Note:``
    is indistinguishable from a single-key YAML front-matter line (which
    must keep working for Jekyll/Hugo docs like ``title: My Post``).
    Multi-word prose before the colon does not match, so such blocks are
    preserved.
    """
    # Known limitation: a bare identifier before the colon looks like YAML,
    # so this block is stripped even though it is prose.
    doc = "---\nNote: important\n---\n\nBody."
    result = _strip_yaml_frontmatter(doc)
    assert "Note:" not in result
    assert "Body" in result

    # Multi-word prose before the colon is not a valid YAML key → preserved.
    doc2 = "---\nSome long prose: with a colon\n---\n\nBody."
    result2 = _strip_yaml_frontmatter(doc2)
    assert "Some long prose" in result2
    assert "Body" in result2


def test_strip_yaml_frontmatter_hostile_whitespace_is_fast() -> None:
    """A '---' opener followed by huge whitespace runs must not blow up.

    The front-matter regexes use [ \\t]* (not \\s*) on delimiter lines:
    with \\s* this input caused quadratic regex backtracking (minutes of
    CPU for a few hundred KB of newlines).
    """
    hostile = "---" + "\n" * 500_000  # opener, no closing delimiter
    start = time.monotonic()
    result = _strip_yaml_frontmatter(hostile)
    assert time.monotonic() - start < 5.0
    assert result == hostile  # nothing stripped

    # Whitespace-only block: matches the block shape but has no YAML keys.
    hostile2 = "---\n" + " \n" * 500_000 + "---\nbody"
    start = time.monotonic()
    result2 = _strip_yaml_frontmatter(hostile2)
    assert time.monotonic() - start < 5.0
    assert result2.endswith("body")


def test_md_crlf_frontmatter_stripped() -> None:
    """CRLF line endings in Markdown must not prevent front-matter stripping.

    TOKENS splitter for the same reason as
    test_md_bytes_with_doc_type_strips_frontmatter: the MARKDOWN chunker
    would strip the front-matter downstream, masking a regression here.
    """
    parser = Parser(ParsingConfig(splitter=Splitter.TOKENS))
    crlf_md = (
        "---\r\ntitle: CRLF Doc\r\nauthor: Bob\r\n---\r\n\r\n# Heading\r\n\r\nBody."
    )
    chunks = DocumentParser.chunks_from_path_or_bytes(
        crlf_md.encode(), parser, doc_type="md"
    )
    full_text = " ".join(c.content for c in chunks)
    assert "title:" not in full_text
    assert "Heading" in full_text


# ---------------------------------------------------------------------------
# DocumentParser.create() raises for MD / HTML
# ---------------------------------------------------------------------------


def test_create_raises_for_md() -> None:
    with pytest.raises(ValueError, match="chunks_from_path_or_bytes"):
        DocumentParser.create(_MD_PATH, ParsingConfig())


def test_create_raises_for_html() -> None:
    with pytest.raises(ValueError, match="chunks_from_path_or_bytes"):
        DocumentParser.create(_HTML_PATH, ParsingConfig())


# ---------------------------------------------------------------------------
# Plain-text bytes ingestion must not regress when magic is unavailable
# ---------------------------------------------------------------------------


def test_txt_bytes_ingest_when_magic_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain-text bytes must ingest as TXT even if `import magic` fails.

    Simulates environments where python-magic is installed but the native
    libmagic library is missing/broken (import raises), by poisoning
    sys.modules so any `import magic` raises ImportError.
    """
    monkeypatch.setitem(sys.modules, "magic", None)

    raw = b"Just some plain text.\nAnother line of text."
    assert DocumentParser._document_type(raw) == DocumentType.TXT

    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser)
    assert len(chunks) > 0
    full_text = " ".join(c.content for c in chunks)
    assert "plain text" in full_text


def test_txt_bytes_ingest_when_magic_broken_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain-text bytes must ingest as TXT if magic raises when called."""
    fake_magic = types.ModuleType("magic")

    def _broken_from_buffer(*args: object, **kwargs: object) -> str:
        raise OSError("failed to find libmagic")

    fake_magic.from_buffer = _broken_from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", fake_magic)

    raw = b"Hello, this is ordinary text content."
    assert DocumentParser._document_type(raw) == DocumentType.TXT

    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser)
    assert len(chunks) > 0
    assert "ordinary text" in " ".join(c.content for c in chunks)


def test_txt_bytes_ingest_when_magic_raises_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANY Exception from magic at call time must trigger the fallback.

    The contract requires surviving any Exception, not just OSError or
    ImportError: a regression narrowing the except clause to specific
    exception types would break this test.
    """
    fake_magic = types.ModuleType("magic")

    def _broken_from_buffer(*args: object, **kwargs: object) -> str:
        raise RuntimeError("magic exploded at call time")

    fake_magic.from_buffer = _broken_from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", fake_magic)

    # Valid UTF-8 bytes must still ingest as TXT.
    raw = b"Generic-failure plain text content."
    assert DocumentParser._document_type(raw) == DocumentType.TXT

    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser)
    assert len(chunks) > 0
    assert "Generic-failure" in " ".join(c.content for c in chunks)

    # Non-text bytes must raise ValueError (not RuntimeError).
    binary = b"\x00\xff\xfe\x00\x01binary-garbage\xff"
    with pytest.raises(ValueError, match="Unsupported document type from bytes"):
        DocumentParser._document_type(binary)


def test_binary_bytes_when_magic_unavailable_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-text bytes with magic unavailable raise ValueError, not ImportError."""
    monkeypatch.setitem(sys.modules, "magic", None)

    binary = b"\x00\xff\xfe\x00\x01binary-garbage\xff"
    with pytest.raises(ValueError, match="Unsupported document type from bytes"):
        DocumentParser._document_type(binary)


@pytest.mark.parametrize("binary", [b"abc\xff", b"\x80"])
def test_trailing_invalid_utf8_bytes_when_magic_unavailable_raise_value_error(
    monkeypatch: pytest.MonkeyPatch, binary: bytes
) -> None:
    """Invalid trailing UTF-8 bytes must not be truncated into TXT."""
    monkeypatch.setitem(sys.modules, "magic", None)

    with pytest.raises(ValueError, match="Unsupported document type from bytes"):
        DocumentParser._document_type(binary)

    parser = Parser(ParsingConfig())
    with pytest.raises(ValueError, match="Unsupported document type from bytes"):
        DocumentParser.chunks_from_path_or_bytes(binary, parser)


def test_incomplete_trailing_utf8_when_magic_unavailable_ingests_as_txt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-boundary UTF-8 prefix is still accepted as plain text."""
    monkeypatch.setitem(sys.modules, "magic", None)
    raw = b"hello \xe2\x82"
    assert DocumentParser._document_type(raw) == DocumentType.TXT

    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser)
    assert len(chunks) > 0
    assert "hello" in " ".join(c.content for c in chunks)


def test_pdf_like_utf8_bytes_when_magic_unavailable_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid-UTF-8 binary signatures must not fall back to TXT."""
    monkeypatch.setitem(sys.modules, "magic", None)
    pdf = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"

    with pytest.raises(ValueError, match="Unsupported document type from bytes"):
        DocumentParser._document_type(pdf)


# ---------------------------------------------------------------------------
# URL sources: .md / .html / .htm URLs must be fetched over HTTP(S)
# ---------------------------------------------------------------------------


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler without per-request stderr logging."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        """Serve special transport and encoding cases used by URL tests."""
        if self.path in {"/stalled", "/stalled.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            time.sleep(0.5)
            return
        if self.path == "/oversized.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            for _ in range(4):
                self.wfile.write(b"0123456789")
                self.wfile.flush()
            return
        if self.path == "/latin1-header.html":
            body = "<p>café déjà vu</p>".encode("iso-8859-1")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=iso-8859-1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/windows1252-meta.html":
            html = (
                '<html><head><meta charset="windows-1252"></head>'
                "<body>Crème brûlée — déjà vu</body></html>"
            )
            body = html.encode("windows-1252")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/utf16-bom.html":
            body = "<p>Snowman ☃</p>".encode("utf-16")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


@pytest.fixture(scope="module")
def doc_server_url(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """Serve copies of the sample docs over local HTTP; yield the base URL."""
    serve_dir = tmp_path_factory.mktemp("docserve")
    shutil.copy(_MD_PATH, serve_dir / "sample.md")
    shutil.copy(_HTML_PATH, serve_dir / "sample.html")
    shutil.copy(_HTML_PATH, serve_dir / "sample.htm")
    handler = functools.partial(_QuietHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_md_url_chunks(doc_server_url: str) -> None:
    """A .md URL must be fetched over HTTP and parsed as Markdown.

    Regression test: extension-first detection classifies the URL as MD,
    and chunks_from_path_or_bytes must fetch it rather than attempt
    open() on the URL string (which raised FileNotFoundError).
    """
    parser = Parser(ParsingConfig())
    url = f"{doc_server_url}/sample.md"
    chunks = DocumentParser.chunks_from_path_or_bytes(url, parser)
    assert len(chunks) > 0
    full_text = "\n".join(c.content for c in chunks)
    assert "Heading One" in full_text
    assert "**Markdown**" in full_text  # raw Markdown preserved
    assert "title:" not in full_text  # front-matter stripped
    assert all(c.metadata.source == url for c in chunks)


def test_md_url_chunks_with_query_and_fragment(doc_server_url: str) -> None:
    parser = Parser(ParsingConfig())
    url = f"{doc_server_url}/sample.md?version=1#intro"
    chunks = DocumentParser.chunks_from_path_or_bytes(url, parser)
    full_text = "\n".join(c.content for c in chunks)
    assert "**Markdown**" in full_text
    assert "title:" not in full_text


@pytest.mark.parametrize(
    "filename",
    [
        "sample.html",
        "sample.htm",
        "sample.html?raw=1",
        "sample.htm?raw=1#section",
    ],
)
def test_html_url_chunks(doc_server_url: str, filename: str) -> None:
    """.html / .htm URLs must be fetched over HTTP and tag-stripped."""
    parser = Parser(ParsingConfig())
    url = f"{doc_server_url}/{filename}"
    chunks = DocumentParser.chunks_from_path_or_bytes(url, parser)
    assert len(chunks) > 0
    full_text = "\n".join(c.content for c in chunks)
    assert "Heading One" in full_text
    assert "<h1>" not in full_text
    # Adjacent elements are newline-separated on the URL path too.
    assert "Item A\nItem B" in full_text


@pytest.mark.parametrize("scheme", ["HTTP", "HtTp"])
def test_url_scheme_is_case_insensitive(doc_server_url: str, scheme: str) -> None:
    """URI schemes are case-insensitive for detection and fetching."""
    url = doc_server_url.replace("http", scheme, 1) + "/sample.html"
    assert DocumentParser._document_type(url) == DocumentType.HTML

    chunks = DocumentParser.chunks_from_path_or_bytes(url, Parser(ParsingConfig()))
    assert "Heading One" in "\n".join(chunk.content for chunk in chunks)


def test_url_read_timeout_stops_stalled_response(doc_server_url: str) -> None:
    """A server that stalls after headers must not hang ingestion."""
    parser = Parser(ParsingConfig(url_read_timeout=0.01))

    with pytest.raises(requests.exceptions.RequestException):
        DocumentParser.chunks_from_path_or_bytes(
            f"{doc_server_url}/stalled.html", parser
        )


def test_configured_timeout_applies_during_url_type_detection(
    doc_server_url: str,
) -> None:
    """Extensionless URL sampling must use the configured read timeout."""
    parser = Parser(ParsingConfig(url_read_timeout=0.01))
    start = time.monotonic()

    with pytest.raises(requests.exceptions.RequestException):
        DocumentParser.chunks_from_path_or_bytes(f"{doc_server_url}/stalled", parser)

    assert time.monotonic() - start < 0.2


def test_url_streaming_rejects_oversized_response(doc_server_url: str) -> None:
    """The streaming size cap applies without a Content-Length header."""
    parser = Parser(ParsingConfig(url_max_size=16))

    with pytest.raises(ValueError, match="maximum.*16 bytes"):
        DocumentParser.chunks_from_path_or_bytes(
            f"{doc_server_url}/oversized.html", parser
        )


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("latin1-header.html", "café déjà vu"),
        ("windows1252-meta.html", "Crème brûlée — déjà vu"),
        ("utf16-bom.html", "Snowman ☃"),
    ],
)
def test_html_url_respects_declared_charset(
    doc_server_url: str, filename: str, expected: str
) -> None:
    """HTTP and HTML charset declarations preserve exact Unicode text."""
    chunks = DocumentParser.chunks_from_path_or_bytes(
        f"{doc_server_url}/{filename}", Parser(ParsingConfig())
    )
    assert expected in "\n".join(chunk.content for chunk in chunks)
