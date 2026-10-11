from io import BytesIO
from pathlib import Path
from typing import List, Optional

import fitz
import pytest

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


@pytest.mark.parametrize("as_stream", [False, True])
@pytest.mark.parametrize(
    "splits, expected_sizes",
    [
        ([2, 2], [2, 2]),
        ([4, 4], [4]),
        ([1, 1], [1, 3]),
        ([3, 1, 3, 1], [1, 2, 1]),
        ([0, 2, 2, 5], [2, 2]),
        ([3, 1], [1, 2, 1]),
        ([], [4]),
        (None, [1, 1, 1, 1]),
    ],
)
def test_pdf_split_pages_preserves_page_sequence(
    tmp_path: Path,
    as_stream: bool,
    splits: Optional[List[int]],
    expected_sizes: List[int],
) -> None:
    """Repeated boundaries must not create reversed ranges or repeat pages."""
    source = tmp_path / "numbered.pdf"
    with fitz.open() as doc:
        for number in range(1, 5):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {number}")
        doc.save(str(source))

    input_pdf = BytesIO(source.read_bytes()) if as_stream else str(source)
    paths, tmp_dir = pdf_split_pages(input_pdf, splits=splits)
    try:
        sizes = []
        texts = []
        for path in paths:
            with fitz.open(str(path)) as part:
                sizes.append(len(part))
                texts.extend(page.get_text().strip() for page in part)
        assert sizes == expected_sizes
        assert texts == [f"Page {number}" for number in range(1, 5)]
        assert len(set(paths)) == len(paths)
    finally:
        tmp_dir.cleanup()
