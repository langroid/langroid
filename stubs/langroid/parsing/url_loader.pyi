from _typeshed import Incomplete

from langroid.mytypes import DocMetaData as DocMetaData
from langroid.mytypes import Document as Document
from langroid.parsing.document_parser import (
    DocumentParser as DocumentParser,
)
from langroid.parsing.document_parser import (
    ImagePdfParser as ImagePdfParser,
)
from langroid.parsing.parser import Parser as Parser
from langroid.parsing.parser import ParsingConfig as ParsingConfig

class URLLoader:
    urls: Incomplete
    parser: Incomplete
    def __init__(self, urls: list[str], parser: Parser = ...) -> None: ...
    def load(self): ...
