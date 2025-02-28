from pathlib import Path

import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import GeminiConfig, ParsingConfig, PdfParsingConfig


@pytest.mark.parametrize("split_on_page", [True, False])
@pytest.mark.parametrize("pdf_file", ["imagenet.pdf"])
def test_gemini_doc_parser(pdf_file, split_on_page):
    current_dir = Path(__file__).resolve().parent
    path = current_dir.parent / "main" / "data" / pdf_file

    parsing_config = ParsingConfig(
        n_neighbor_ids=2,
        pdf=PdfParsingConfig(
            library="gemini",
            gemini_config=GeminiConfig(
                model_name="gemini-2.0-flash",
                split_on_page=split_on_page,
            ),
        ),
    )

    gemini_parser = DocumentParser.create(
        path.as_posix(),
        parsing_config,
    )
    doc = gemini_parser.get_doc()
    pages = [page for page in gemini_parser.iterate_pages()]

    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty

    assert (
        "with magnitudes proportional to the corresponding eigenvalues"
        in pages[0][1][:70].strip()
    )
    if split_on_page:
        assert "obvious in static images." in pages[2][1][-50:].replace(
            "\n", ""
        ).replace("8", "")
    else:
        assert "obvious in static images." in pages[0][1][-50:].replace(
            "\n", ""
        ).replace("8", "")
    assert doc.metadata.source == str(path)

    docs = gemini_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    n = len(docs)
    k = gemini_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1
