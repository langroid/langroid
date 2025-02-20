from pathlib import Path

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import GeminiConfig, ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("pdf_file", ["imagenet.pdf"])
def test_gemini_doc_parser(tmp_path, pdf_file):
    current_dir = Path(__file__).resolve().parent
    path = current_dir.parent / "main" / "data" / pdf_file
    output_file = tmp_path / "parsed_docs.md"  # Use temporary path

    parsing_config = ParsingConfig(
        n_neighbor_ids=2,
        pdf=PdfParsingConfig(
            library="gemini",
            gemini_config=GeminiConfig(
                model_name="gemini-2.0-flash",
                split_on_page=True,
                output_filename=output_file.as_posix(),
            ),
        ),
    )

    gemini_parser = DocumentParser.create(
        path.as_posix(),
        parsing_config,
    )
    doc = gemini_parser.get_doc()

    # Assertions
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == str(path)

    docs = gemini_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    n = len(docs)
    k = gemini_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1
