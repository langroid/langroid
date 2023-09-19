import os

from langroid.parsing.parser import Parser, ParsingConfig
from langroid.parsing.pdf_parser import PdfParser


def test_get_pdf_doc_url():
    url = "https://arxiv.org/pdf/2104.05490.pdf"
    doc = PdfParser.get_doc_from_pdf_url(url)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == url

    parser = Parser(ParsingConfig())
    pdfParser = PdfParser.from_Parser(parser)
    docs = pdfParser.doc_chunks_from_pdf_url(url, parser)
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)


def test_get_pdf_doc_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the PDF file
    path = os.path.join(current_dir, "test.pdf")
    doc = PdfParser.get_doc_from_pdf_file(path)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path

    parser = Parser(ParsingConfig())
    pdfParser = PdfParser.from_Parser(parser)
    docs = pdfParser.doc_chunks_from_pdf_path(path, parser)
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
