from langroid.parsing.pdf_utils import pdf_split_pages


def test_pdf_split_pages():
    # Test with a sample PDF file of 4 pages
    pdf_path = "tests/main/data/dummy.pdf"
    pages, _ = pdf_split_pages(pdf_path)

    # Check if the pages are split correctly
    assert len(pages) == 4

    parts, _ = pdf_split_pages(pdf_path, splits=[3])
    assert len(parts) == 2

    parts, _ = pdf_split_pages(pdf_path, splits=[1, 2])
    assert len(parts) == 3
