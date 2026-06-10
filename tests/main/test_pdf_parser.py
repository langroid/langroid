import os

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("source", ["url", "bytes"])
@pytest.mark.parametrize(
    "pdflib",
    [
        "docling",
        "fitz",
        "pypdf",
        "pypdfium2",
        "unstructured",
        "pymupdf4llm",
        pytest.param(
            "marker",
            marks=pytest.mark.xfail(
                reason=(
                    "marker-pdf pins openai<2 / litellm<1.83, which conflict "
                    "with langroid's security-patched lock (openai 2.x, "
                    "litellm 1.83+, transformers 4.57); the resulting locked "
                    "marker 1.6.1 is too old for transformers 4.57. Users who "
                    "install langroid[marker] get a working marker via normal "
                    "pip resolution."
                ),
                strict=False,
            ),
        ),
    ],
)
def test_get_pdf_doc_url(source, pdflib: str):
    url = "tests/main/data/openr-1-3.pdf"
    pdf_parser = DocumentParser.create(
        url,
        ParsingConfig(
            n_neighbor_ids=2,
            pdf=PdfParsingConfig(library=pdflib),
        ),
    )

    if source == "bytes":
        bytes = pdf_parser._load_doc_as_bytesio()
        pdf_parser = DocumentParser.create(
            bytes.getvalue(), pdf_parser.config  # convert BytesIO to bytes
        )

    doc = pdf_parser.get_doc()

    # PdfParser.get_doc_from_pdf_url(url)

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == ("bytes" if source == "bytes" else url)

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


@pytest.mark.xfail(
    condition=lambda pdflib: pdflib == "marker",
    reason="Marker may timeout",
    strict=False,
)
@pytest.mark.parametrize("source", ["path", "bytes"])
@pytest.mark.parametrize(
    "pdflib",
    ["unstructured", "docling", "fitz", "pypdf", "pypdfium2", "pymupdf4llm", "marker"],
)
def test_get_pdf_doc_path(source, pdflib: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))
    path = os.path.join(tests_root, "main", "data", "dummy.pdf")

    pdf_parser = DocumentParser.create(
        path, ParsingConfig(pdf=PdfParsingConfig(library=pdflib))
    )

    if source == "bytes":
        with open(path, "rb") as f:
            bytes = f.read()
        pdf_parser = DocumentParser.create(bytes, pdf_parser.config)

    doc = pdf_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    citation = path if source == "path" else "bytes"
    assert doc.metadata.source == citation

    docs = pdf_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    assert all(citation in d.metadata.source for d in docs)


def test_default_pdf_library_is_permissive():
    """
    The default PDF parser must be the permissively-licensed `pypdfium2`
    (Apache-2.0/BSD-3-Clause), NOT the AGPL-licensed `pymupdf4llm`, since
    Langroid is MIT-licensed and the default parser ships in core deps.
    See issue #1026.
    """
    assert PdfParsingConfig().library == "pypdfium2"


@pytest.mark.parametrize("source", ["path", "bytes"])
def test_pypdfium2_parser(source: str):
    """
    Dedicated functional test for the default `pypdfium2` PDF parser.

    `pypdfium2` is installed by default (core dependency), so this test does
    NOT require any optional `extras` to be installed -- it exercises the
    out-of-the-box PDF-parsing path that a bare `pip install langroid` gets.
    """
    from langroid.parsing.document_parser import PyPDFium2Parser

    path = "tests/main/data/dummy.pdf"
    parser = DocumentParser.create(
        path, ParsingConfig(pdf=PdfParsingConfig(library="pypdfium2"))
    )
    assert isinstance(parser, PyPDFium2Parser)

    if source == "bytes":
        with open(path, "rb") as f:
            data = f.read()
        parser = DocumentParser.create(data, parser.config)
        assert isinstance(parser, PyPDFium2Parser)

    citation = path if source == "path" else "bytes"

    doc = parser.get_doc()
    assert isinstance(doc.content, str)
    # content correctness: known text from the sample PDF
    assert "Design and Evaluation" in doc.content
    assert "arXiv:2004.07606v1" in doc.content
    assert doc.metadata.source == citation

    chunks = parser.get_doc_chunks()
    assert len(chunks) > 0
    assert all(c.metadata.is_chunk for c in chunks)
    assert all(citation in c.metadata.source for c in chunks)


# @pytest.mark.skipif(
#     os.environ.get("CI") == "true",
#     reason="GH Actions/Ubuntu has issues with pdf2image/pyteseract",
# )


@pytest.mark.parametrize("source", ["url", "bytes"])
@pytest.mark.parametrize(
    "path",
    [
        "https://nlsblog.org/wp-content/uploads/2020/06/image-based-pdf-sample.pdf",
        "tests/main/data/image-based-pdf-sample.pdf",
    ],
)
def test_image_pdf(source, path):
    """
    Test text extraction from an image-pdf
    """
    cfg = ParsingConfig(pdf=PdfParsingConfig(library="pdf2image"))
    pdf_parser = DocumentParser.create(path, cfg)
    doc = pdf_parser.get_doc()
    if source == "bytes":
        bytes = pdf_parser._load_doc_as_bytesio()
        pdf_parser = DocumentParser.create(bytes.getvalue(), cfg)

    doc = pdf_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    citation = path if source == "url" else "bytes"
    assert doc.metadata.source == citation

    docs = pdf_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)

    assert all(citation in d.metadata.source for d in docs)
