import os

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import GeminiParsingConfig, ParsingConfig


def test_gemini_doc_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))

    path = os.path.join(tests_root, "main", "data", "imagenet.pdf")
    gemini_parser = DocumentParser.create(
        path,
        ParsingConfig(
            n_neighbor_ids=2,
            gemini=GeminiParsingConfig(gemini_model_name="gemini-2.0-flash"),
        ),
    )
    doc = gemini_parser.get_doc()

    # Check the results
    assert isinstance(doc.content, str)
    assert len(doc.content) > 0  # assuming the PDF is not empty
    assert doc.metadata.source == path

    docs = gemini_parser.get_doc_chunks()
    assert len(docs) > 0
    assert all(d.metadata.is_chunk for d in docs)
    n = len(docs)
    k = gemini_parser.config.n_neighbor_ids
    if n > 2 * k + 1:
        assert len(docs[n // 2].metadata.window_ids) == 2 * k + 1
