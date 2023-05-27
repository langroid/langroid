from pydantic import BaseModel, Extra
from typing import Union, List

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]


class DocMetaData(BaseModel):
    """Metadata for a document."""

    source: str = "context"

    class Config:
        extra = Extra.allow


class Document(BaseModel):
    """Interface for interacting with a document."""

    content: str
    metadata: DocMetaData

    def __str__(self):
        # TODO: make metadata a pydantic model to enforce "source"
        self.metadata.json()
        return f"{self.content} {self.metadata.json()}"
