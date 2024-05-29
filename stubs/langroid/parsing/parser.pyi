from enum import Enum
from typing import Literal

from _typeshed import Incomplete
from pydantic import BaseSettings

from langroid.mytypes import Document as Document
from langroid.parsing.para_sentence_split import (
    create_chunks as create_chunks,
)
from langroid.parsing.para_sentence_split import (
    remove_extra_whitespace as remove_extra_whitespace,
)

logger: Incomplete

class Splitter(str, Enum):
    TOKENS: str
    PARA_SENTENCE: str
    SIMPLE: str

class PdfParsingConfig(BaseSettings):
    library: Literal["fitz", "pdfplumber", "pypdf", "unstructured", "pdf2image"]

class DocxParsingConfig(BaseSettings):
    library: Literal["python-docx", "unstructured"]

class DocParsingConfig(BaseSettings):
    library: Literal["unstructured"]

class ParsingConfig(BaseSettings):
    splitter: str
    chunk_size: int
    overlap: int
    max_chunks: int
    min_chunk_chars: int
    discard_chunk_chars: int
    n_similar_docs: int
    n_neighbor_ids: int
    separators: list[str]
    token_encoding_model: str
    pdf: PdfParsingConfig
    docx: DocxParsingConfig
    doc: DocParsingConfig

class Parser:
    config: Incomplete
    tokenizer: Incomplete
    def __init__(self, config: ParsingConfig) -> None: ...
    def num_tokens(self, text: str) -> int: ...
    def add_window_ids(self, chunks: list[Document]) -> None: ...
    def split_simple(self, docs: list[Document]) -> list[Document]: ...
    def split_para_sentence(self, docs: list[Document]) -> list[Document]: ...
    def split_chunk_tokens(self, docs: list[Document]) -> list[Document]: ...
    def chunk_tokens(self, text: str) -> list[str]: ...
    def split(self, docs: list[Document]) -> list[Document]: ...
