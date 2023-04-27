from dataclasses import dataclass
from llmagent.parsing.config import ParsingConfig
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llmagent.mytypes import Document
from langchain.schema import Document as LDocument
from typing import List

@dataclass
class Parser(ParsingConfig):
    def num_tokens(self, text:str) -> int:
        encoding = tiktoken.encoding_for_model(self.token_encoding_model)
        return len(encoding.encode(text))

    def split(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.num_tokens,
        )
        # list of LangChain Documents
        lc_docs = [LDocument(page_content = d.content, metadata = d.metadata)
                   for d in docs]
        texts = text_splitter.split_documents(lc_docs)

        # convert texts to list of Documents
        texts = [Document(content=text.page_content, metadata=text.metadata)
                 for text in texts]
        return texts





