"""
Tests for Markdown and HTML support in DocumentParser / DocumentType.

Covers:
- DocumentType enum has MD and HTML values
- _document_type() detects .md, .html, .htm extensions correctly
- DocumentType.MD correctly precedes is_plain_text() heuristic
- chunks_from_path_or_bytes() returns non-empty chunks for both formats
- Markdown YAML front-matter is stripped
- HTML tags are stripped; readable text is preserved
- DocumentParser.create() raises a clear ValueError for MD/HTML
  (they are handled via chunks_from_path_or_bytes, not via a subclass)
"""

import os

import pytest

from langroid.parsing.document_parser import (
    DocumentParser,
    DocumentType,
    _strip_yaml_frontmatter,
)
from langroid.parsing.parser import Parser, ParsingConfig

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
    ],
)
def test_document_type_extension_detection(
    filename: str, expected: DocumentType
) -> None:
    # Use _document_type directly; pass a fake path with the right extension.
    # The file doesn't need to exist — we only care about extension dispatch.
    result = DocumentParser._document_type(filename)
    assert result == expected


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


def test_md_chunks_from_path() -> None:
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(_MD_PATH, parser)

    assert len(chunks) > 0
    full_text = " ".join(c.content for c in chunks)

    # Front-matter should be stripped
    assert "title:" not in full_text
    assert "author:" not in full_text

    # Body content must survive
    assert "Heading One" in full_text
    assert "Item A" in full_text
    assert "blockquote" in full_text or "blockquote paragraph" in full_text


def test_md_chunks_from_bytes() -> None:
    with open(_MD_PATH, "rb") as f:
        raw = f.read()
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="md")

    assert len(chunks) > 0
    full_text = " ".join(c.content for c in chunks)
    assert "Heading One" in full_text


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


def test_md_bytes_with_doc_type_strips_frontmatter() -> None:
    """Markdown bytes passed with doc_type='md' must strip YAML front-matter."""
    with open(_MD_PATH, "rb") as f:
        raw = f.read()
    parser = Parser(ParsingConfig())
    chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="md")
    full_text = " ".join(c.content for c in chunks)
    assert "title:" not in full_text
    assert "Heading One" in full_text


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


def test_strip_yaml_frontmatter_preserves_thematic_break() -> None:
    """A leading --- used as a thematic break (no key: value inside) must not be removed."""
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


def test_md_crlf_frontmatter_stripped() -> None:
    """CRLF line endings in Markdown must not prevent front-matter stripping."""
    parser = Parser(ParsingConfig())
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
