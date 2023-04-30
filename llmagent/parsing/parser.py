from pydantic import BaseSettings
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llmagent.mytypes import Document
from langchain.schema import Document as LDocument
from typing import List


class ParsingConfig (BaseSettings):
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
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.config.separators,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self.num_tokens,
        )
        # list of LangChain Documents
        lc_docs = [LDocument(page_content=d.content, metadata=d.metadata) for d in docs]
        texts = text_splitter.split_documents(lc_docs)

        # convert texts to list of Documents
        texts = [
            Document(content=text.page_content, metadata=text.metadata)
            for text in texts
        ]
        return texts
