import os

from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import MarkitDownParsingConfig, ParsingConfig

url1 = "tests/main/dummy.pdf"
url2 = "tests/main/iris.xls"
url3 = "tests/main/doc-test-file.doc"
url4 = "docs/notes/weaviate.md"
url5 = "tests/main/sample.pptx"


def test_markitdown():
    pdf_parser = DocumentParser.create(
        url1,
        ParsingConfig(
            n_neighbor_ids=2,
            pdf=MarkitDownParsingConfig(),
        ),
        doc_type="pdf",
    )
    doc = pdf_parser.get_doc()
    print(doc)
    print("_" * 40)
    # little hack to set doc_type of xls to xlsx as markitdown has problems with xls
    # since markitdown processes many types we have to set the doc_type so that
    # it processes the stream reliably
    xlsx_parser = DocumentParser.create(
        url2,
        ParsingConfig(
            n_neighbor_ids=2,
            markitdown=MarkitDownParsingConfig(),
        ),
        doc_type="xlsx",
    )

    doc = xlsx_parser.get_doc()
    print(doc)
    print("_" * 40)

    for url in [url3, url4, url5]:
        ext = os.path.splitext(url)[1].lower()[1:]
        parser = DocumentParser.create(
            url,
            ParsingConfig(
                n_neighbor_ids=2,
                markitdown=MarkitDownParsingConfig(),
            ),
            doc_type=ext,
        )
        doc = parser.get_doc()
        print(doc)
        print("_" * 40)


test_markitdown()
