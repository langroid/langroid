from io import BytesIO

import requests
from PyPDF2 import PdfReader

from langroid.mytypes import DocMetaData, Document


def get_pdf_doc_url(url: str) -> Document:
    """
    Args:
        url (str): contains the URL to the PDF file
    Returns:
        a `Document` object containing the content of the pdf file,
        and metadata containing url
    """
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    text = ""
    for page_num in range(len(reader.pages)):
        current_page = reader.pages[page_num]
        text += current_page.extract_text()
    return Document(content=text, metadata=DocMetaData(source=str(url)))


def get_pdf_doc_path(path: str) -> Document:
    """
    Args:
        path (str): full path to the PDF file
        url (str| None): contains the URL of the PDF file if the processed
        PDF file obtained via URL
    Returns:
        a `Document` object containing the content of the pdf file,
        and metadata containing path/url
    """
    reader = PdfReader(path)

    text = ""
    for page_num in range(len(reader.pages)):
        current_page = reader.pages[page_num]
        text += current_page.extract_text()

    return Document(content=text, metadata=DocMetaData(source=str(path)))
