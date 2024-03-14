import logging
import os
from tempfile import NamedTemporaryFile
from typing import List, no_type_check

import requests
import trafilatura
from trafilatura.downloads import (
    add_to_compressed_dict,
    buffered_downloads,
    load_download_buffer,
)

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.document_parser import DocumentParser, ImagePdfParser
from langroid.parsing.parser import Parser, ParsingConfig

logging.getLogger("trafilatura").setLevel(logging.ERROR)


class URLLoader:
    """
    Load a list of URLs and extract the text content.
    Alternative approaches could use `bs4` or `scrapy`.

    TODO - this currently does not handle cookie dialogs,
     i.e. if there is a cookie pop-up, most/all of the extracted
     content could be cookie policy text.
     We could use `playwright` to simulate a user clicking
     the "accept" button on the cookie dialog.
    """

    def __init__(self, urls: List[str], parser: Parser = Parser(ParsingConfig())):
        self.urls = urls
        self.parser = parser

    @no_type_check
    def load(self) -> List[Document]:
        docs = []
        threads = 4
        # converted the input list to an internal format
        dl_dict = add_to_compressed_dict(self.urls)
        # processing loop
        while not dl_dict.done:
            buffer, dl_dict = load_download_buffer(
                dl_dict,
                sleep_time=5,
            )
            for url, result in buffered_downloads(buffer, threads):
                if (
                    url.lower().endswith(".pdf")
                    or url.lower().endswith(".docx")
                    or url.lower().endswith(".doc")
                ):
                    doc_parser = DocumentParser.create(
                        url,
                        self.parser.config,
                    )
                    new_chunks = doc_parser.get_doc_chunks()
                    if len(new_chunks) == 0:
                        # If the document is empty, try to extract images
                        img_parser = ImagePdfParser(url, self.parser.config)
                        new_chunks = img_parser.get_doc_chunks()
                    docs.extend(new_chunks)
                else:
                    # Try to detect content type and handle accordingly
                    headers = requests.head(url).headers
                    content_type = headers.get("Content-Type", "").lower()
                    temp_file_suffix = None
                    if "application/pdf" in content_type:
                        temp_file_suffix = ".pdf"
                    elif (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        in content_type
                    ):
                        temp_file_suffix = ".docx"
                    elif "application/msword" in content_type:
                        temp_file_suffix = ".doc"

                    if temp_file_suffix:
                        # Download the document content
                        response = requests.get(url)
                        with NamedTemporaryFile(
                            delete=False, suffix=temp_file_suffix
                        ) as temp_file:
                            temp_file.write(response.content)
                            temp_file_path = temp_file.name
                        # Process the downloaded document
                        doc_parser = DocumentParser.create(
                            temp_file_path, self.parser.config
                        )
                        docs.extend(doc_parser.get_doc_chunks())
                        # Clean up the temporary file
                        os.remove(temp_file_path)
                    else:
                        text = trafilatura.extract(
                            result,
                            no_fallback=False,
                            favor_recall=True,
                        )
                        if (
                            text is None
                            and result is not None
                            and isinstance(result, str)
                        ):
                            text = result
                        if text is not None and text != "":
                            docs.append(
                                Document(content=text, metadata=DocMetaData(source=url))
                            )
        return docs
