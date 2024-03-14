import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("pdflib", ["fitz", "pypdf", "pdfplumber"])
def test_get_pdf_doc_url(pdflib: str):
    url = "https://arxiv.org/pdf/2104.05490.pdf"
    pdf_parser = DocumentParser.create(
        url,
        ParsingConfig(
            n_neighbor_ids=2,
            pdf=PdfParsingConfig(library=pdflib),
        ),
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
    n = len(docs)
    k = pdf_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1


@pytest.mark.parametrize("pdflib", ["fitz", "pypdf", "pdfplumber"])
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


@pytest.mark.parametrize(
    "path",
    [
        "https://nlsblog.org/wp-content/uploads/2020/06/image-based-pdf-sample.pdf",
        "tests/main/data/image-based-pdf-sample.pdf",
    ],
)
def test_image_pdf(path):
    """
    Test text extraction from an image-pdf
    """
    cfg = ParsingConfig(pdf=PdfParsingConfig(library="pdf2image"))
    pdf_parser = DocumentParser.create(path, cfg)
    doc = pdf_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path

    docs = pdf_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    assert all(path in d.metadata.source for d in docs)
