from functools import reduce
from typing import Callable, List

import tiktoken
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

from langroid.mytypes import Document
from langroid.pydantic_v1 import BaseSettings


def chunk_code(
    code: str, language: str, max_tokens: int, len_fn: Callable[[str], int]
) -> List[str]:
    """
    Chunk code into smaller pieces, so that we don't exceed the maximum
    number of tokens allowed by the embedding model.
    Args:
        code: string of code
        language: str as a file extension, e.g. "py", "yml"
        max_tokens: max tokens per chunk
        len_fn: function to get the length of a string in token units
    Returns:

    """
    lexer = get_lexer_by_name(language)
    tokens = list(lex(code, lexer))

    chunks = []
    current_chunk = ""
    for token_type, token_value in tokens:
        if token_type in Token.Text.Whitespace:
            current_chunk += token_value
        else:
            token_tokens = len_fn(token_value)
            if len_fn(current_chunk) + token_tokens <= max_tokens:
                current_chunk += token_value
            else:
                chunks.append(current_chunk)
                current_chunk = token_value

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class CodeParsingConfig(BaseSettings):
    extensions: List[str] = [
        "py",
        "java",
        "c",
        "cpp",
        "h",
        "hpp",
        "yml",
        "yaml",
        "toml",
        "cfg",  # e.g. setup.cfg
        "ini",
        "json",
        "rst",
        "sh",
        "bash",
    ]
    chunk_size: int = 500  # tokens
    token_encoding_model: str = "text-embedding-ada-002"
    n_similar_docs: int = 4


class CodeParser:
    def __init__(self, config: CodeParsingConfig):
        self.config = config
        self.tokenizer = tiktoken.encoding_for_model(config.token_encoding_model)

    def num_tokens(self, text: str) -> int:
        """
        How many tokens are in the text, according to the tokenizer.
        This needs to be accurate, otherwise we may exceed the maximum
        number of tokens allowed by the model.
        Args:
            text: string to tokenize
        Returns:
            number of tokens in the text
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split the documents into chunks, according to the config.splitter.
        Only the documents with a language in the config.extensions are split.
        !!! note
            We assume the metadata in each document has at least a `language` field,
            which is used to determine how to chunk the code.
        Args:
            docs: list of documents to split
        Returns:
            list of documents, where each document is a chunk; the metadata of the
            original document is duplicated for each chunk, so that when we retrieve a
            chunk, we immediately know info about the original document.
        """
        chunked_docs = [
            [
                Document(content=chunk, metadata=d.metadata)
                for chunk in chunk_code(
                    d.content,
                    d.metadata.language,  # type: ignore
                    self.config.chunk_size,
                    self.num_tokens,
                )
                if chunk.strip() != ""
            ]
            for d in docs
            if d.metadata.language in self.config.extensions  # type: ignore
        ]
        if len(chunked_docs) == 0:
            return []
        # collapse the list of lists into a single list
        return reduce(lambda x, y: x + y, chunked_docs)
