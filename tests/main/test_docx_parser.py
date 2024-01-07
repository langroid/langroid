import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import DocxParsingConfig, ParsingConfig


@pytest.mark.parametrize("docxlib", ["unstructured", "python-docx"])
def test_get_docx_file(docxlib: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "docx-test-file.docx")
    docx_parser = DocumentParser.create(
        path, ParsingConfig(docx=DocxParsingConfig(library=docxlib))
    )
    doc = docx_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path

    # parser = Parser(ParsingConfig())
    # pdfParser = PdfParser.from_Parser(parser)
    # docs = pdfParser.doc_chunks_from_pdf_url(url, parser)
    docs = docx_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    assert all(path in d.metadata.source for d in docs)
