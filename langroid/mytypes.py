from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4

from langroid.pydantic_v1 import BaseModel, Extra, Field

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]
EmbeddingFunction = Callable[[List[str]], Embeddings]


class Entity(str, Enum):
    """
    Enum for the different types of entities that can respond to the current message.
    """

    AGENT = "Agent"
    LLM = "LLM"
    USER = "User"
    SYSTEM = "System"

    def __eq__(self, other: object) -> bool:
        """Allow case-insensitive equality (==) comparison with strings."""
        if other is None:
            return False
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        """Allow case-insensitive non-equality (!=) comparison with strings."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Override this to ensure hashability of the enum,
        so it can be used sets and dictionary keys.
        """
        return hash(self.value.lower())


class DocMetaData(BaseModel):
    """Metadata for a document."""

    source: str = "context"  # just reference
    source_content: str = "context"  # reference and content
    title: str = "Unknown Title"
    published_date: str = "Unknown Date"
    is_chunk: bool = False  # if it is a chunk, don't split
    id: str = Field(default_factory=lambda: str(uuid4()))
    window_ids: List[str] = []  # for RAG: ids of chunks around this one

    def dict_bool_int(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Special dict method to convert bool fields to int, to appease some
        downstream libraries,  e.g. Chroma which complains about bool fields in
        metadata.
        """
        original_dict = super().dict(*args, **kwargs)

        for key, value in original_dict.items():
            if isinstance(value, bool):
                original_dict[key] = 1 * value

        return original_dict

    def __str__(self) -> str:
        title_str = (
            ""
            if "unknown" in self.title.lower() or self.title.strip() == ""
            else f"Title: {self.title}"
        )
        date_str = ""
        if (
            "unknown" not in self.published_date.lower()
            and self.published_date.strip() != ""
        ):
            try:
                from dateutil import parser

                # Try to parse the date string
                date_obj = parser.parse(self.published_date)
                # Format to include only the date part (year-month-day)
                date_only = date_obj.strftime("%Y-%m-%d")
                date_str = f"Date: {date_only}"
            except (ValueError, ImportError, TypeError):
                # If parsing fails, just use the original date
                date_str = f"Date: {self.published_date}"
        components = [self.source] + (
            [] if title_str + date_str == "" else [title_str, date_str]
        )
        return ", ".join(components)

    class Config:
        extra = Extra.allow


class Document(BaseModel):
    """Interface for interacting with a document."""

    content: str
    metadata: DocMetaData

    def id(self) -> str:
        return self.metadata.id

    @staticmethod
    def from_string(
        content: str,
        source: str = "context",
        is_chunk: bool = True,
    ) -> "Document":
        return Document(
            content=content,
            metadata=DocMetaData(source=source, is_chunk=is_chunk),
        )

    def __str__(self) -> str:
        return dedent(
            f"""
        CONTENT: {self.content}         
        SOURCE:{str(self.metadata)}
        """
        )


class NonToolAction(str, Enum):
    """
    Possible options to handle non-tool msgs from LLM.
    """

    FORWARD_USER = "user"  # forward msg to user
    DONE = "done"  # task done
