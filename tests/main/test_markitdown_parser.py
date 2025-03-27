import os

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import (
    MarkitdownPPTXParsingConfig,
    MarkitdownXLSParsingConfig,
    ParsingConfig,
)


def test_markitdown_xls_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))

    path1 = os.path.join(tests_root, "main", "data", "sample.xlsx")

    # Test XLS parsing
    xls_parser = DocumentParser.create(
        path1,
        ParsingConfig(
            n_neighbor_ids=2,
            xls=MarkitdownXLSParsingConfig(),
        ),
    )
    doc_xls = xls_parser.get_doc()
    assert isinstance(doc_xls.content, str)
    assert len(doc_xls.content) > 0
    assert doc_xls.metadata.source == path1

    xls_chunks = xls_parser.get_doc_chunks()
    assert len(xls_chunks) > 0
    assert all(chunk.metadata.is_chunk for chunk in xls_chunks)
    assert all(path1 in chunk.metadata.source for chunk in xls_chunks)


def test_markitdown_pptx_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_root = os.path.abspath(os.path.join(current_dir, ".."))

    path = os.path.join(tests_root, "main", "data", "sample.pptx")

    # Test PPTX parsing
    pptx_parser = DocumentParser.create(
        path,
        ParsingConfig(
            n_neighbor_ids=2,
            pptx=MarkitdownPPTXParsingConfig(),
        ),
    )
    doc_pptx = pptx_parser.get_doc()
    assert isinstance(doc_pptx.content, str)
    assert len(doc_pptx.content) > 0
    assert doc_pptx.metadata.source == path

    pptx_chunks = pptx_parser.get_doc_chunks()
    assert len(pptx_chunks) > 0
    assert all(chunk.metadata.is_chunk for chunk in pptx_chunks)
    assert all(path in chunk.metadata.source for chunk in pptx_chunks)
