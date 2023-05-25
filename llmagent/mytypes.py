from pydantic import BaseModel, Field
from typing import Union, List
import json

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]


class Document(BaseModel):
    """Interface for interacting with a document."""

    content: str
    metadata: dict = Field(default_factory=dict)

    def __str__(self):
        # TODO: make metadata a pydantic model to enforce "source"
        return f"{self.content} {json.dumps(self.metadata)}"
