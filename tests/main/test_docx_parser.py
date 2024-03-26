import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import DocxParsingConfig, ParsingConfig


@pytest.mark.parametrize("source", ["path", "bytes"])
@pytest.mark.parametrize("docxlib", ["python-docx"])
def test_get_docx_file(source, docxlib: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "docx-test-file.docx")
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
