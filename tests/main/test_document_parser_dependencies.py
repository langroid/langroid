"""Missing optional backends must raise actionable import errors."""

import builtins
from typing import Any

import pytest

from langroid.exceptions import LangroidImportError
from langroid.parsing.document_parser import DocumentParser, DocumentType
from langroid.parsing.parser import DocxParsingConfig, ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize(
    "doc_type, package, extra",
    [
        (DocumentType.PDF, "fitz", "doc-chat"),
        (DocumentType.DOCX, "markitdown", "markitdown"),
        (DocumentType.XLSX, "markitdown", "doc-parsers"),
        (DocumentType.PPTX, "markitdown", "doc-parsers"),
    ],
)
@pytest.mark.parametrize("method", ["get_doc", "get_doc_chunks"])
def test_missing_document_backend_has_installation_hint(
    monkeypatch: pytest.MonkeyPatch,
    doc_type: DocumentType,
    package: str,
    extra: str,
    method: str,
) -> None:
    parser = DocumentParser.create(
        b"",
        ParsingConfig(
            pdf=PdfParsingConfig(library="fitz"),
            docx=DocxParsingConfig(library="markitdown-docx"),
        ),
        doc_type=doc_type,
    )
    original_import = builtins.__import__

    def import_without_backend(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == package:
            raise ModuleNotFoundError(f"No module named '{package}'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_backend)
    with pytest.raises(LangroidImportError) as exc:
        getattr(parser, method)()
    assert package in str(exc.value)
    assert f'pip install "langroid[{extra}]"' in str(exc.value)
