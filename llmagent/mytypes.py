from pydantic import BaseModel, Field
from typing import Union, List

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]


class Document(BaseModel):
    """Interface for interacting with a document."""

    content: str
    metadata: dict = Field(default_factory=dict)
