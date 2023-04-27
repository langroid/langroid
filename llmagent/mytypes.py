from pydantic import BaseModel, Field


class Document(BaseModel):
    """Interface for interacting with a document."""
    content: str
    metadata: dict = Field(default_factory=dict)

