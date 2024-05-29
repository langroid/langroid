from _typeshed import Incomplete

from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document

def accept_cookies_and_extract_content(url: str) -> str: ...

class URLLoader:
    urls: Incomplete
    def __init__(self, urls: list[str]) -> None: ...
    def load(self): ...
