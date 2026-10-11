import ast
from pathlib import Path

import pytest

import langroid.parsing.document_parser as document_parser
from langroid.parsing.document_parser import DocumentParser, DocumentType
from langroid.parsing.parser import Parser, ParsingConfig, Splitter


@pytest.mark.parametrize("source_kind", ["path", "bytes", "url"])
@pytest.mark.parametrize("doc_type", [DocumentType.TXT, DocumentType.MD])
@pytest.mark.parametrize("limited", [False, True])
def test_document_line_reads_preserve_whitespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    doc_type: DocumentType,
    limited: bool,
) -> None:
    code = "def greet():\n    return 'hello'  "
    text = f"```python\n{code}\n```" if doc_type == DocumentType.MD else code
    raw = (text + "\nignored()").encode()
    if source_kind == "path":
        path = tmp_path / f"source.{doc_type.value}"
        path.write_bytes(raw)
        source: str | bytes = str(path)
    elif source_kind == "url":
        source = f"https://example.test/source.{doc_type.value}"
        monkeypatch.setattr(
            document_parser, "fetch_configured_url", lambda *args: (raw, {})
        )
    else:
        source = raw

    parser = Parser(
        ParsingConfig(splitter=Splitter.TOKENS, chunk_size=1000, discard_chunk_chars=0)
    )
    chunks = DocumentParser.chunks_from_path_or_bytes(
        source,
        parser,
        doc_type=doc_type,
        lines=len(text.splitlines()) if limited else None,
    )
    assert len(chunks) == 1
    assert chunks[0].content == (text if limited else text + "\nignored()")
    if doc_type == DocumentType.TXT:
        ast.parse(chunks[0].content)


@pytest.mark.parametrize("limited", [False, True])
def test_html_line_reads_preserve_preformatted_indentation(
    tmp_path: Path, limited: bool
) -> None:
    path = tmp_path / "source.html"
    path.write_text("<pre>def greet():\n    return 'hello'  \n</pre>")
    parser = Parser(
        ParsingConfig(splitter=Splitter.TOKENS, chunk_size=1000, discard_chunk_chars=0)
    )
    chunks = DocumentParser.chunks_from_path_or_bytes(
        str(path), parser, lines=3 if limited else None
    )
    assert len(chunks) == 1
    assert chunks[0].content == "def greet():\n    return 'hello'  \n"
    ast.parse(chunks[0].content)
