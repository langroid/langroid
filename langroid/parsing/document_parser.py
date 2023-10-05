import logging
import re
from enum import Enum
from io import BytesIO
from typing import Any, Generator, List, Tuple

import fitz
import pdfplumber
import pypdf
import requests

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.parsing.urls import url_to_tempfile

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"


class DocumentParser(Parser):
    """
    Abstract base class for extracting text from special types of docs
    such as PDFs or Docx.

    Attributes:
        source (str): The source, either a URL or a file path.
        doc_bytes (BytesIO): BytesIO object containing the doc data.
    """

    @classmethod
    def create(cls, source: str, config: ParsingConfig) -> "DocumentParser":
        """
        Create a DocumentParser instance based on source type
            and config.<source_type>.library specified.

        Args:
            source (str): The source of the PDF, either a URL or a file path.
            config (ParserConfig): The parser configuration.

        Returns:
            DocumentParser: An instance of a DocumentParser subclass.
        """
        if DocumentParser._document_type(source) == DocumentType.PDF:
            if config.pdf.library == "fitz":
                return FitzPDFParser(source, config)
            elif config.pdf.library == "pypdf":
                return PyPDFParser(source, config)
            elif config.pdf.library == "pdfplumber":
                return PDFPlumberParser(source, config)
            elif config.pdf.library == "unstructured":
                return UnstructuredPDFParser(source, config)
            elif config.pdf.library == "haystack":
                return HaystackPDFParser(source, config)
            else:
                raise ValueError(
                    f"Unsupported PDF library specified: {config.pdf.library}"
                )
        elif DocumentParser._document_type(source) == DocumentType.DOCX:
            if config.docx.library == "unstructured":
                return UnstructuredDocxParser(source, config)
            else:
                raise ValueError(
                    f"Unsupported DOCX library specified: {config.docx.library}"
                )
        else:
            raise ValueError(f"Unsupported document type: {source}")

    def __init__(self, source: str, config: ParsingConfig):
        """
        Initialize the PDFParser.

        Args:
            source (str): The source of the PDF, either a URL or a file path.
        """
        super().__init__(config)
        self.source = source
        self.config = config
        self.doc_bytes = self._load_doc_as_bytesio()

    @staticmethod
    def _document_type(source: str) -> DocumentType:
        """
        Determine the type of document based on the source.

        Args:
            source (str): The source of the PDF, either a URL or a file path.

        Returns:
            str: The document type.
        """
        if source.lower().endswith(".pdf"):
            return DocumentType.PDF
        elif source.lower().endswith(".docx"):
            return DocumentType.DOCX
        else:
            raise ValueError(f"Unsupported document type: {source}")

    def _load_doc_as_bytesio(self) -> BytesIO:
        """
        Load the docs into a BytesIO object.

        Returns:
            BytesIO: A BytesIO object containing the doc data.
        """
        if self.source.startswith(("http://", "https://")):
            response = requests.get(self.source)
            response.raise_for_status()
            return BytesIO(response.content)
        else:
            with open(self.source, "rb") as f:
                return BytesIO(f.read())

    def iterate_pages(self) -> Generator[Tuple[int, Any], None, None]:
        """Yield each page in the PDF."""
        raise NotImplementedError

    def extract_text_from_page(self, page: Any) -> str:
        """Extract text from a given page."""
        raise NotImplementedError

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
            page_text = self.extract_text_from_page(page)
            split += self.tokenizer.encode(page_text)
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


class FitzPDFParser(DocumentParser):
    """
    Parser for processing PDFs using the `fitz` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, fitz.Page], None, None]:
        """
        Yield each page in the PDF using `fitz`.

        Returns:
            Generator[fitz.Page]: Generator yielding each page.
        """
        doc = fitz.open(stream=self.doc_bytes, filetype="pdf")
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


class PyPDFParser(DocumentParser):
    """
    Parser for processing PDFs using the `pypdf` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, pypdf.PageObject], None, None]:
        """
        Yield each page in the PDF using `pypdf`.

        Returns:
            Generator[pypdf.pdf.PageObject]: Generator yielding each page.
        """
        reader = pypdf.PdfReader(self.doc_bytes)
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


class PDFPlumberParser(DocumentParser):
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
        with pdfplumber.open(self.doc_bytes) as pdf:
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


class HaystackPDFParser(DocumentParser):
    """
    Parser for processing PDFs using the `haystack` library.
    """

    def get_doc_chunks(self) -> List[Document]:
        """
        Overrides the base class method to use the `haystack` library.
        See there for more details.
        """

        from haystack.nodes import PDFToTextConverter, PreProcessor

        converter = PDFToTextConverter(
            remove_numeric_tables=True,
        )
        path = self.source
        if path.startswith(("http://", "https://")):
            path = url_to_tempfile(path)
        doc = converter.convert(file_path=path, meta=None)
        # note self.config.chunk_size is in token units,
        # and we use an approximation of 75 words per 100 tokens
        # to convert to word units
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=int(0.75 * self.config.chunk_size),
            split_overlap=int(0.75 * self.config.overlap),
            split_respect_sentence_boundary=True,
            add_page_number=True,
        )
        chunks = preprocessor.process(doc)
        return [
            Document(
                content=chunk.content,
                metadata=DocMetaData(
                    source=f"{self.source} page {chunk.meta['page']}",
                    is_chunk=True,
                ),
            )
            for chunk in chunks
        ]


class UnstructuredPDFParser(DocumentParser):
    """
    Parser for processing PDF files using the `unstructured` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, Any], None, None]:  # type: ignore
        from unstructured.partition.pdf import partition_pdf

        # from unstructured.chunking.title import chunk_by_title

        try:
            elements = partition_pdf(file=self.doc_bytes, include_page_breaks=True)
        except Exception as e:
            raise Exception(
                f"""
                Error parsing PDF: {e}
                The `unstructured` library failed to parse the pdf.
                Please try a different library by setting the `library` field
                in the `pdf` section of the `parsing` field in the config file.
                Supported libraries are: 
                fitz, pypdf, pdfplumber, unstructured, haystack 
                """
            )

        # elements = chunk_by_title(elements)
        page_number = 1
        page_elements = []  # type: ignore
        for el in elements:
            if el.category == "PageBreak":
                if page_elements:  # Avoid yielding empty pages at the start
                    yield page_number, page_elements
                page_number += 1
                page_elements = []
            else:
                page_elements.append(el)
        # Yield the last page if it's not empty
        if page_elements:
            yield page_number, page_elements

    def extract_text_from_page(self, page: Any) -> str:
        """
        Extract text from a given `unstructured` element.

        Args:
            page (unstructured element): The `unstructured` element object.

        Returns:
            str: Extracted text from the element.
        """
        text = " ".join(el.text for el in page)
        return self.fix_text(text)


class UnstructuredDocxParser(DocumentParser):
    """
    Parser for processing DOCX files using the `unstructured` library.
    """

    def iterate_pages(self) -> Generator[Tuple[int, Any], None, None]:  # type: ignore
        from unstructured.partition.docx import partition_docx

        elements = partition_docx(file=self.doc_bytes, include_page_breaks=True)

        page_number = 1
        page_elements = []  # type: ignore
        for el in elements:
            if el.category == "PageBreak":
                if page_elements:  # Avoid yielding empty pages at the start
                    yield page_number, page_elements
                page_number += 1
                page_elements = []
            else:
                page_elements.append(el)
        # Yield the last page if it's not empty
        if page_elements:
            yield page_number, page_elements

    def extract_text_from_page(self, page: Any) -> str:
        """
        Extract text from a given `unstructured` element.

        Note:
            The concept of "pages" doesn't actually exist in the .docx file format in
            the same way it does in formats like .pdf. A .docx file is made up of a
            series of elements like paragraphs and tables, but the division into
            pages is done dynamically based on the rendering settings (like the page
            size, margin size, font size, etc.).

        Args:
            page (unstructured element): The `unstructured` element object.

        Returns:
            str: Extracted text from the element.
        """
        text = " ".join(el.text for el in page)
        return self.fix_text(text)
