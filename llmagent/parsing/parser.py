from pydantic import BaseSettings
import tiktoken
from functools import reduce
from llmagent.parsing.para_sentence_split import create_chunks
from llmagent.mytypes import Document
from typing import List


class ParsingConfig(BaseSettings):
    splitter: str = "para_sentence"
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = ["\n\n", "\n", " ", ""]
    token_encoding_model: str = "text-davinci-003"


class Parser:
    def __init__(self, config: ParsingConfig):
        self.config = config

    def num_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.config.token_encoding_model)
        return len(encoding.encode(text))

    def split(self, docs: List[Document]) -> List[Document]:
        if self.config.splitter == "para_sentence":
            return self.split_para_sentence(docs)
        else:
            raise ValueError(f"Unknown splitter: {self.config.splitter}")

    def split_para_sentence(self, docs: List[Document]) -> List[Document]:
        chunked_docs = [
            [
                Document(content=chunk.strip(), metadata=d.metadata)
                for chunk in create_chunks(
                    d.content, self.config.chunk_size, self.num_tokens
                )
                if chunk.strip() != ""
            ]
            for d in docs
        ]
        return reduce(lambda x, y: x + y, chunked_docs)
