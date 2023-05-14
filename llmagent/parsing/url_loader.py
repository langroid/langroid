from typing import List
from llmagent.mytypes import Document
import trafilatura
from trafilatura.downloads import (
    add_to_compressed_dict,
    buffered_downloads,
    load_download_buffer,
)
import logging

logging.getLogger("trafilatura").setLevel(logging.ERROR)


class URLLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load(self) -> List[Document]:
        docs = []
        threads = 4
        backoff_dict = dict()  # has to be defined first
        # converted the input list to an internal format
        dl_dict = add_to_compressed_dict(self.urls)
        # processing loop
        while dl_dict:
            buffer, dl_dict = load_download_buffer(
                dl_dict,
                backoff_dict,
            )
            if len(buffer) == 0:
                break
            for url, result in buffered_downloads(buffer, threads):
                text = trafilatura.extract(result, no_fallback=True)
                if text is not None:
                    docs.append(Document(content=text, metadata={"source": url}))
        return docs
