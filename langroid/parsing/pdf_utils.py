import tempfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, List, Tuple, Union

try:
    import fitz
except ImportError:
    if not TYPE_CHECKING:
        fitz = None

from langroid.exceptions import LangroidImportError

if fitz is None:
    raise LangroidImportError("fitz", ["pymupdf", "all", "pdf-parsers", "doc-chat"])


def pdf_split_pages(
    input_pdf: Union[BytesIO, BinaryIO],
) -> Tuple[List[Path], TemporaryDirectory[Any]]:
    """Splits a PDF into individual pages in a temporary directory.

    Args:
        input_pdf: Input PDF file in bytes or binary mode
        max_workers: Maximum number of concurrent workers for parallel processing

    Returns:
        Tuple containing:
            - List of paths to individual PDF pages
            - Temporary directory object (caller must call cleanup())

    Example:
        paths, tmp_dir = split_pdf_temp("input.pdf")
        # Use paths...
        tmp_dir.cleanup()  # Clean up temp files when done
    """
    tmp_dir = tempfile.TemporaryDirectory()
    doc = fitz.open(stream=input_pdf, filetype="pdf")
    paths = []

    for page_num in range(len(doc)):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        output = Path(tmp_dir.name) / f"page_{page_num + 1}.pdf"
        new_doc.save(str(output))
        new_doc.close()
        paths.append(output)

    doc.close()
    return paths, tmp_dir
