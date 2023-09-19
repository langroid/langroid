from io import BytesIO
from typing import List

import requests
from pypdf import PdfReader

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import Parser


class PdfParser(Parser):
    def __init__(self, parser: Parser):
        super().__init__(parser.config)

    @classmethod
    def from_Parser(cls, parser: Parser) -> "PdfParser":
        return cls(parser)

    @staticmethod
    def _text_from_pdf_reader(reader: PdfReader) -> str:
        """
        Extract text from a `PdfReader` object.
        Args:
            reader (PdfReader): a `PdfReader` object
        Returns:
            str: the extracted text
        """
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _doc_chunks_from_pdf_reader(
        self,
        reader: PdfReader,
        doc: str,
        chunk_tokens: int,
        overlap: int = 0,
    ) -> List[Document]:
        """
        Get document chunks from a PdfReader object,
        with page references in the document metadata.

        Adapted from
        https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py

        Args:
            reader (PdfReader): a `PdfReader` object
            doc: URL or filename of the PDF file
            chunk_tokens (int): number of tokens in each chunk
            overlap (int): number of tokens to overlap between chunks

        Returns:
            List[Document]: a list of `Document` objects,
                each containing a chunk of text
        """

        split = []  # tokens in curr split
        pages: List[str] = []
        docs: List[Document] = []
        for i, page in enumerate(reader.pages):
            split += self.tokenizer.encode(page.extract_text())
            pages.append(str(i + 1))
            # split could be so long it needs to be split
            # into multiple chunks. Or it could be so short
            # that it needs to be combined with the next chunk.
            while len(split) > chunk_tokens:
                # pretty formatting of pages (e.g. 1-3, 4, 5-7)
                pg = "-".join([pages[0], pages[-1]])
                docs.append(
                    Document(
                        content=self.tokenizer.decode(split[:chunk_tokens]),
                        metadata=DocMetaData(
                            source=f"{doc} pages {pg}",
                            is_chunk=True,
                        ),
                    )
                )
                split = split[chunk_tokens - overlap :]
                pages = [str(i + 1)]
        if len(split) > overlap:
            pg = "-".join([pages[0], pages[-1]])
            docs.append(
                Document(
                    content=self.tokenizer.decode(split[:chunk_tokens]),
                    metadata=DocMetaData(
                        source=f"{doc} pages {pg}",
                        is_chunk=True,
                    ),
                )
            )
        return docs

    @staticmethod
    def doc_chunks_from_pdf_url(url: str, parser: Parser) -> List[Document]:
        """
        Get chunks of text from pdf URL as a list of Document objects,
        using the parser's chunk_size and overlap settings.

        Args:
            url (str): contains the URL to the PDF file
        Returns:
            a `Document` object containing the content of the pdf file,
                and metadata containing url
        """

        pdfParser = PdfParser.from_Parser(parser)
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            docs = pdfParser._doc_chunks_from_pdf_reader(
                reader,
                doc=url,
                chunk_tokens=parser.config.chunk_size,
                overlap=parser.config.overlap,
            )
        return docs

    @staticmethod
    def get_doc_from_pdf_url(url: str) -> Document:
        """
        Get entire text from pdf URL as a single document.

        Args:
            url (str): contains the URL to the PDF file
        Returns:
            a `Document` object containing the content of the pdf file,
                and metadata containing url
        """
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = PdfParser._text_from_pdf_reader(reader)
        return Document(content=text, metadata=DocMetaData(source=str(url)))

    @staticmethod
    def doc_chunks_from_pdf_path(path: str, parser: Parser) -> List[Document]:
        """
        Get chunks of text from pdf path as a list of Document objects,
        using the parser's chunk_size and overlap settings.

        Args:
            url (str): contains the URL to the PDF file
        Returns:
            a `Document` object containing the content of the pdf file,
                and metadata containing url
        """

        pdfParser = PdfParser.from_Parser(parser)
        reader = PdfReader(path)
        docs = pdfParser._doc_chunks_from_pdf_reader(
            reader,
            doc=path,
            chunk_tokens=parser.config.chunk_size,
            overlap=parser.config.overlap,
        )
        return docs

    @staticmethod
    def get_doc_from_pdf_file(path: str) -> Document:
        """
        Given local path to a PDF file, extract the text content.
        Args:
            path (str): full path to the PDF file
                PDF file obtained via URL
        Returns:
            a `Document` object containing the content of the pdf file,
                and metadata containing path/url
        """
        reader = PdfReader(path)
        text = PdfParser._text_from_pdf_reader(reader)
        return Document(content=text, metadata=DocMetaData(source=str(path)))
