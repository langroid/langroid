import os

from langroid.parsing.pdf_parser import (
    get_pdf_doc_path,
    get_pdf_doc_url,
)


def test_get_pdf_doc_url():
    url = "https://arxiv.org/pdf/2104.05490.pdf"
    doc = get_pdf_doc_url(url)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == url


def test_get_pdf_doc_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the PDF file
    path = os.path.join(current_dir, "dummy.pdf")
    doc = get_pdf_doc_path(path)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path
