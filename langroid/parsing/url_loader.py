import logging
from typing import List, no_type_check

import trafilatura
from trafilatura.downloads import (
    add_to_compressed_dict,
    buffered_downloads,
    load_download_buffer,
)

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.pdf_parser import get_doc_from_pdf_url

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

    def __init__(self, urls: List[str]):
        self.urls = urls

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
                if ".pdf" in url:
                    docs.append(get_doc_from_pdf_url(url))
                else:
                    text = trafilatura.extract(
                        result,
                        no_fallback=False,
                        favor_recall=True,
                    )
                    if text is not None and text != "":
                        docs.append(
                            Document(content=text, metadata=DocMetaData(source=url))
                        )
        return docs
