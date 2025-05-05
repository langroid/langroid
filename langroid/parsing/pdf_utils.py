import tempfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, List, Optional, Tuple, Union

try:
    import fitz
except ImportError:
    if not TYPE_CHECKING:
        fitz = None

from langroid.exceptions import LangroidImportError

if fitz is None:
    raise LangroidImportError("fitz", ["pymupdf", "all", "pdf-parsers", "doc-chat"])


def pdf_split_pages(
    input_pdf: Union[BytesIO, BinaryIO, str],
    splits: Optional[List[int]] = None,
) -> Tuple[List[Path], TemporaryDirectory[Any]]:
    """Splits a PDF into individual pages or chunks in a temporary directory.

    Args:
        input_pdf: Input PDF file in bytes, binary mode, or a file path
        splits: Optional list of page numbers to split at.
                If provided, pages will be grouped into chunks ending at
                these page numbers.
                For example, if splits = [4, 9], the result will have pages 1-4, 5-9,
                and 10-end.
                If not provided, default to splitting into individual pages.
        max_workers: Maximum number of concurrent workers for parallel processing

    Returns:
        Tuple containing:
            - List of paths to individual PDF pages or chunks
            - Temporary directory object (caller must call cleanup())

    Example:
        paths, tmp_dir = split_pdf_temp("input.pdf")
        # Use paths...
        tmp_dir.cleanup()  # Clean up temp files when done
    """
    tmp_dir = tempfile.TemporaryDirectory()
    if isinstance(input_pdf, str):
        doc = fitz.open(input_pdf)
    else:
        doc = fitz.open(stream=input_pdf, filetype="pdf")
    paths = []

    total_pages = len(doc)

    if splits is None:
        # Split into individual pages (original behavior)
        for page_num in range(total_pages):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            output = Path(tmp_dir.name) / f"page_{page_num + 1}.pdf"
            new_doc.save(str(output))
            new_doc.close()
            paths.append(output)
    else:
        # Split according to specified page ranges
        # Make sure the splits list is sorted and includes all valid splits
        splits = sorted([s for s in splits if 1 <= s <= total_pages])

        # Create the ranges to process
        ranges = []
        start_page = 0
        for end_page in splits:
            ranges.append((start_page, end_page - 1))
            start_page = end_page

        # Add the final range if there are pages after the last split
        if start_page < total_pages:
            ranges.append((start_page, total_pages - 1))

        # Process each range
        for i, (from_page, to_page) in enumerate(ranges):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=from_page, to_page=to_page)
            output = Path(tmp_dir.name) / f"pages_{from_page + 1}_to_{to_page + 1}.pdf"
            new_doc.save(str(output))
            new_doc.close()
            paths.append(output)

    doc.close()
    return paths, tmp_dir
