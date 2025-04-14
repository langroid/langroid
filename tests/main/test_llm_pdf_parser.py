from pathlib import Path

import nest_asyncio
import pytest

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import LLMPdfParserConfig, ParsingConfig, PdfParsingConfig
from langroid.utils.configuration import Settings, set_global

nest_asyncio.apply()


@pytest.mark.asyncio
@pytest.mark.parametrize("split_on_page", [True, False])
@pytest.mark.parametrize("pdf_file", ["imagenet.pdf"])
async def test_llm_pdf_parser(pdf_file, split_on_page):
    # disable `chat_model` setting so it doesn't interfere with mdl below
    set_global(Settings(chat_model=""))
    current_dir = Path(__file__).resolve().parent
    path = current_dir.parent / "main" / "data" / pdf_file

    parsing_config = ParsingConfig(
        n_neighbor_ids=2,
        pdf=PdfParsingConfig(
            library="llm-pdf-parser",
            llm_parser_config=LLMPdfParserConfig(
                model_name="gemini/gemini-2.0-flash",
                split_on_page=split_on_page,
            ),
        ),
    )

    llm_parser = DocumentParser.create(
        path.as_posix(),
        parsing_config,
    )
    doc = llm_parser.get_doc()
    pages = [page for page in llm_parser.iterate_pages()]

    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty

    assert (
        "with magnitudes proportional to the corresponding eigenvalues"
        in pages[0][1][:80].strip()
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

    docs = llm_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    n = len(docs)
    k = llm_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1
