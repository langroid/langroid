import tempfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, BinaryIO, List, Tuple, Union

try:
    import pypdf
except ImportError:
    if not TYPE_CHECKING:
        pypdf = None

from langroid.exceptions import LangroidImportError

if pypdf is None:
    raise LangroidImportError(
        "pypdf", ["pypdf", "docling", "all", "pdf-parsers", "doc-chat"]
    )
from pypdf import PdfReader, PdfWriter


def pdf_split_pages(
    input_pdf: Union[str, Path, BytesIO, BinaryIO],
) -> Tuple[List[Path], TemporaryDirectory[Any]]:
    """Splits a PDF into individual pages in a temporary directory.

    Args:
        input_pdf: Input PDF file path or file-like object
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
    reader = PdfReader(input_pdf)
    paths = []

    for i in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        writer.add_metadata(reader.metadata or {})

        output = Path(tmp_dir.name) / f"page_{i+1}.pdf"
        with open(output, "wb") as f:
            writer.write(f)
        paths.append(output)

    return paths, tmp_dir  # Return dir object so caller can control cleanup
