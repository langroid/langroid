from pathlib import Path

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("pdf_file", ["imagenet.pdf"])
def test_marker_pdf_parser(pdf_file):
    current_dir = Path(__file__).resolve().parent
    path = current_dir.parent / "main" / "data" / pdf_file

    parsing_config = ParsingConfig(
        n_neighbor_ids=2,
        pdf=PdfParsingConfig(
            library="marker",
        ),
    )

    marker_parser = DocumentParser.create(
        path.as_posix(),
        parsing_config,
    )
    doc = marker_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == str(path)
    docs = marker_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    n = len(docs)
    k = marker_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1
