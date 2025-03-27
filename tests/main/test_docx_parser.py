import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import DocxParsingConfig, ParsingConfig


@pytest.mark.parametrize("source", ["path", "bytes"])
@pytest.mark.parametrize("docxlib", ["python-docx"])
def test_get_docx_file(source, docxlib: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))
    path = os.path.join(tests_root, "main", "data", "docx-test-file.docx")
    docx_parser = DocumentParser.create(
        path, ParsingConfig(docx=DocxParsingConfig(library=docxlib))
    )
    if source == "bytes":
        bytes = docx_parser._load_doc_as_bytesio()
        docx_parser = DocumentParser.create(
            bytes.getvalue(), docx_parser.config  # convert BytesIO to bytes
        )
    doc = docx_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the docx is not empty
    citation = path if source == "path" else "bytes"
    assert doc.metadata.source == citation

    # parser = Parser(ParsingConfig())
    # pdfParser = PdfParser.from_Parser(parser)
    # docs = pdfParser.doc_chunks_from_pdf_url(url, parser)
    docs = docx_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    assert all(citation in d.metadata.source for d in docs)


def test_markitdown_docx_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))

    path = os.path.join(tests_root, "main", "data", "sample.docx")

    # Test DOCX parsing
    docx_parser = DocumentParser.create(
        path,
        ParsingConfig(
            n_neighbor_ids=2,
            docx=DocxParsingConfig(library="markitdown-docx"),
        ),
    )
    doc_docx = docx_parser.get_doc()
    assert isinstance(doc_docx.content, str)
    assert len(doc_docx.content) > 0
    assert doc_docx.metadata.source == path

    docx_chunks = docx_parser.get_doc_chunks()
    assert len(docx_chunks) > 0
    assert all(chunk.metadata.is_chunk for chunk in docx_chunks)
    assert all(path in chunk.metadata.source for chunk in docx_chunks)
