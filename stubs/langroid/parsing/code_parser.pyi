from typing import Callable

from _typeshed import Incomplete
from pydantic import BaseSettings

from langroid.mytypes import Document as Document

def chunk_code(
    code: str, language: str, max_tokens: int, len_fn: Callable[[str], int]
) -> list[str]: ...

class CodeParsingConfig(BaseSettings):
    extensions: list[str]
    chunk_size: int
    token_encoding_model: str
    n_similar_docs: int

class CodeParser:
    config: Incomplete
    tokenizer: Incomplete
    def __init__(self, config: CodeParsingConfig) -> None: ...
    def num_tokens(self, text: str) -> int: ...
    def split(self, docs: list[Document]) -> list[Document]: ...
