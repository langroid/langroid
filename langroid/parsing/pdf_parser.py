import re
from abc import abstractmethod
from io import BytesIO
from typing import Any, Generator, List, Tuple

import fitz
import pdfplumber
import pypdf
import requests

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import Parser, ParsingConfig


class PdfParser(Parser):
    """
    Abstract base class for extracting text from PDFs.

    Attributes:
        source (str): The PDF source, either a URL or a file path.
        pdf_bytes (BytesIO): BytesIO object containing the PDF data.
    """

    @classmethod
    def create(cls, source: str, config: ParsingConfig) -> "PdfParser":
        """
        Create a PDF Parser instance based on config.library specified.

        Args:
            source (str): The source of the PDF, either a URL or a file path.
            config (ParserConfig): The parser configuration.

        Returns:
            PdfParser: An instance of a PDF Parser subclass.
        """
        if config.pdf.library == "fitz":
            return FitzPdfParser(source, config)
        elif config.pdf.library == "pypdf":
            return PyPdfParser(source, config)
        elif config.pdf.library == "pdfplumber":
            return PdfPlumberParser(source, config)
        else:
            raise ValueError(f"Unsupported library specified: {config.pdf.library}")

    def __init__(self, source: str, config: ParsingConfig):
        """
        Initialize the PDFParser.

        Args:
            source (str): The source of the PDF, either a URL or a file path.
        """
        super().__init__(config)
        self.source = source
        self.config = config
        self.pdf_bytes = self._load_pdf_as_bytesio()

    def _load_pdf_as_bytesio(self) -> BytesIO:
        """
        Load the PDF into a BytesIO object.

        Returns:
            BytesIO: A BytesIO object containing the PDF data.
        """
        if self.source.startswith(("http://", "https://")):
            response = requests.get(self.source)
            response.raise_for_status()
            return BytesIO(response.content)
        else:
            with open(self.source, "rb") as f:
                return BytesIO(f.read())

    @abstractmethod
    def iterate_pages(self) -> Generator[Tuple[int, Any], None, None]:
        """Yield each page in the PDF."""
        pass

    @abstractmethod
    def extract_text_from_page(self, page: Any) -> str:
        """Extract text from a given page."""
        pass

    def fix_text(self, text: str) -> str:
        """
        Fix text extracted from a PDF.

        Args:
            text (str): The extracted text.

        Returns:
            str: The fixed text.
        """
        # Some pdf parsers introduce extra space before hyphen,
        # so use regular expression to replace 'space-hyphen' with just 'hyphen'
        return re.sub(r" +\-", "-", text)

    def get_doc(self) -> Document:
        """
        Get entire text from pdf source as a single document.

        Returns:
            a `Document` object containing the content of the pdf file,
                and metadata containing source name (URL or path)
        """

        text = "".join(
            [self.extract_text_from_page(page) for _, page in self.iterate_pages()]
        )
        return Document(content=text, metadata=DocMetaData(source=self.source))

    def get_doc_chunks(self) -> List[Document]:
        """
        Get document chunks from a pdf source,
        with page references in the document metadata.

        Adapted from
        https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py

        Returns:
            List[Document]: a list of `Document` objects,
                each containing a chunk of text
        """

        split = []  # tokens in curr split
        pages: List[str] = []
        docs: List[Document] = []
        for i, page in self.iterate_pages():
            split += self.tokenizer.encode(self.extract_text_from_page(page))
            pages.append(str(i + 1))
            # split could be so long it needs to be split
            # into multiple chunks. Or it could be so short
            # that it needs to be combined with the next chunk.
            while len(split) > self.config.chunk_size:
                # pretty formatting of pages (e.g. 1-3, 4, 5-7)
                pg = "-".join([pages[0], pages[-1]])
                text = self.tokenizer.decode(split[: self.config.chunk_size])
                docs.append(
                    Document(
                        content=text,
                        metadata=DocMetaData(
                            source=f"{self.source} pages {pg}",
                            is_chunk=True,
                        ),
                    )
                )
                split = split[self.config.chunk_size - self.config.overlap :]
                pages = [str(i + 1)]
        if len(split) > self.config.overlap:
            pg = "-".join([pages[0], pages[-1]])
            text = self.tokenizer.decode(split[: self.config.chunk_size])
            docs.append(
                Document(
                    content=text,
                    metadata=DocMetaData(
                        source=f"{self.source} pages {pg}",
                        is_chunk=True,
                    ),
                )
            )
        return docs


class FitzPdfParser(PdfParser):
    """
    Parser for processing PDFs using the `fitz` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, fitz.Page], None, None]:
        """
        Yield each page in the PDF using `fitz`.

        Returns:
            Generator[fitz.Page]: Generator yielding each page.
        """
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            yield i, page
        doc.close()

    def extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a given `fitz` page.

        Args:
            page (fitz.Page): The `fitz` page object.

        Returns:
            str: Extracted text from the page.
        """
        return self.fix_text(page.get_text())


class PyPdfParser(PdfParser):
    """
    Parser for processing PDFs using the `pypdf` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, pypdf.PageObject], None, None]:
        """
        Yield each page in the PDF using `pypdf`.

        Returns:
            Generator[pypdf.pdf.PageObject]: Generator yielding each page.
        """
        reader = pypdf.PdfReader(self.pdf_bytes)
        for i, page in enumerate(reader.pages):
            yield i, page

    def extract_text_from_page(self, page: pypdf.PageObject) -> str:
        """
        Extract text from a given `pypdf` page.

        Args:
            page (pypdf.pdf.PageObject): The `pypdf` page object.

        Returns:
            str: Extracted text from the page.
        """
        return self.fix_text(page.extract_text())


class PdfPlumberParser(PdfParser):
    """
    Parser for processing PDFs using the `pdfplumber` library.
    """

    def iterate_pages(
        self,
    ) -> (Generator)[Tuple[int, pdfplumber.pdf.Page], None, None]:  # type: ignore
        """
        Yield each page in the PDF using `pdfplumber`.

        Returns:
            Generator[pdfplumber.Page]: Generator yielding each page.
        """
        with pdfplumber.open(self.pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                yield i, page

    def extract_text_from_page(self, page: pdfplumber.pdf.Page) -> str:  # type: ignore
        """
        Extract text from a given `pdfplumber` page.

        Args:
            page (pdfplumber.Page): The `pdfplumber` page object.

        Returns:
            str: Extracted text from the page.
        """
        return self.fix_text(page.extract_text())
