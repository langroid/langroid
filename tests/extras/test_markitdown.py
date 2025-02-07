import os

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import (
    MarkitdownPPTXParsingConfig,
    MarkitdownXLSParsingConfig,
    ParsingConfig,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
tests_root = os.path.abspath(os.path.join(current_dir, ".."))
path1 = os.path.join(tests_root, "main", "data", "iris.xls")
path2 = os.path.join(tests_root, "main", "data", "sample.pptx")


def test_markitdown():
    xls_parser = DocumentParser.create(
        path1,
        ParsingConfig(
            n_neighbor_ids=2,
            xls=MarkitdownXLSParsingConfig(),
        ),
    )

    doc = xls_parser.get_doc()
    print(doc)
    print("_" * 40)

    pptx_parser = DocumentParser.create(
        path2,
        ParsingConfig(
            n_neighbor_ids=2,
            pptx=MarkitdownPPTXParsingConfig(),
        ),
    )

    doc = pptx_parser.get_doc()
    print(doc)
    print("_" * 40)


test_markitdown()
