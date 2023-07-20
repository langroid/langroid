from io import BytesIO

import requests
from pypdf import PdfReader

from langroid.mytypes import DocMetaData, Document


def _text_from_pdf_reader(reader: PdfReader) -> str:
    """
    Extract text from a `PdfReader` object.
    Args:
        reader (PdfReader): a `PdfReader` object
    Returns:
        str: the extracted text
    """
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def get_doc_from_pdf_url(url: str) -> Document:
    """
    Args:
        url (str): contains the URL to the PDF file
    Returns:
        a `Document` object containing the content of the pdf file,
            and metadata containing url
    """
    response = requests.get(url)
    response.raise_for_status()
    with BytesIO(response.content) as f:
        reader = PdfReader(f)
        text = _text_from_pdf_reader(reader)
    return Document(content=text, metadata=DocMetaData(source=str(url)))


def get_doc_from_pdf_file(path: str) -> Document:
    """
    Given local path to a PDF file, extract the text content.
    Args:
        path (str): full path to the PDF file
            PDF file obtained via URL
    Returns:
        a `Document` object containing the content of the pdf file,
            and metadata containing path/url
    """
    reader = PdfReader(path)
    text = _text_from_pdf_reader(reader)
    return Document(content=text, metadata=DocMetaData(source=str(path)))
