import logging
from typing import List, no_type_check

import trafilatura
from trafilatura.downloads import (
    add_to_compressed_dict,
    buffered_downloads,
    load_download_buffer,
)

from llmagent.mytypes import DocMetaData, Document

logging.getLogger("trafilatura").setLevel(logging.ERROR)


class URLLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls

    @no_type_check
    def load(self) -> List[Document]:
        docs = []
        threads = 4
        backoff_dict = dict()
        # converted the input list to an internal format
        dl_dict = add_to_compressed_dict(self.urls)
        # processing loop
        while dl_dict:
            buffer, dl_dict = load_download_buffer(
                dl_dict,
                backoff_dict,
            )
            if dl_dict.done:
                break
            for url, result in buffered_downloads(buffer, threads):
                text = trafilatura.extract(result, no_fallback=True)
                if text is not None:
                    docs.append(
                        Document(content=text, metadata=DocMetaData(source=url))
                    )
        return docs
