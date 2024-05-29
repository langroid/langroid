from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import parser
    from . import agent_chats
    from . import code_parser
    from . import document_parser
    from . import parse_json
    from . import para_sentence_split
    from . import repo_loader
    from . import url_loader
    from . import table_loader
    from . import urls
    from . import utils
    from . import search
    from . import web_search
    from . import spider

    from .parser import (
        Splitter,
        PdfParsingConfig,
        DocxParsingConfig,
        DocParsingConfig,
        ParsingConfig,
        Parser,
    )
else:
    parser = LazyLoad("langroid.parsing.parser")
    agent_chats = LazyLoad("langroid.parsing.agent_chats")
    code_parser = LazyLoad("langroid.parsing.code_parser")
    document_parser = LazyLoad("langroid.parsing.document_parser")
    parse_json = LazyLoad("langroid.parsing.parse_json")
    para_sentence_split = LazyLoad("langroid.parsing.para_sentence_split")
    repo_loader = LazyLoad("langroid.parsing.repo_loader")
    url_loader = LazyLoad("langroid.parsing.url_loader")
    table_loader = LazyLoad("langroid.parsing.table_loader")
    urls = LazyLoad("langroid.parsing.urls")
    utils = LazyLoad("langroid.parsing.utils")
    search = LazyLoad("langroid.parsing.search")
    web_search = LazyLoad("langroid.parsing.web_search")
    spider = LazyLoad("langroid.parsing.spider")

    Splitter = LazyLoad("langroid.parsing.parser.Splitter")
    PdfParsingConfig = LazyLoad("langroid.parsing.parser.PdfParsingConfig")
    DocxParsingConfig = LazyLoad("langroid.parsing.parser.DocxParsingConfig")
    DocParsingConfig = LazyLoad("langroid.parsing.parser.DocParsingConfig")
    ParsingConfig = LazyLoad("langroid.parsing.parser.ParsingConfig")
    Parser = LazyLoad("langroid.parsing.parser.Parser")

__all__ = [
    "parser",
    "agent_chats",
    "code_parser",
    "document_parser",
    "parse_json",
    "para_sentence_split",
    "repo_loader",
    "url_loader",
    "table_loader",
    "urls",
    "utils",
    "search",
    "web_search",
    "spider",
    "Splitter",
    "PdfParsingConfig",
    "DocxParsingConfig",
    "DocParsingConfig",
    "ParsingConfig",
    "Parser",
]
