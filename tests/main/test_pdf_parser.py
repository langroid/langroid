import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("pdflib", ["fitz", "pypdf", "pdfplumber", "unstructured"])
def test_get_pdf_doc_url(pdflib: str):
    url = "https://arxiv.org/pdf/2104.05490.pdf"
    pdf_parser = DocumentParser.create(
        url, ParsingConfig(pdf=PdfParsingConfig(library=pdflib))
    )
    doc = pdf_parser.get_doc()
    # PdfParser.get_doc_from_pdf_url(url)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == url

    # parser = Parser(ParsingConfig())
    # pdfParser = PdfParser.from_Parser(parser)
    # docs = pdfParser.doc_chunks_from_pdf_url(url, parser)
    docs = pdf_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)


@pytest.mark.parametrize("pdflib", ["fitz", "pypdf", "pdfplumber", "unstructured"])
def test_get_pdf_doc_path(pdflib: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the PDF file
    path = os.path.join(current_dir, "dummy.pdf")
    pdf_parser = DocumentParser.create(
        path, ParsingConfig(pdf=PdfParsingConfig(library=pdflib))
    )
    doc = pdf_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path

    docs = pdf_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    assert all(path in d.metadata.source for d in docs)
