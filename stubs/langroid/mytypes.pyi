from enum import Enum
from typing import Any, Callable

from _typeshed import Incomplete
from pydantic import BaseModel

Number = int | float
Embedding = list[Number]
Embeddings = list[Embedding]
EmbeddingFunction = Callable[[list[str]], Embeddings]

class Entity(str, Enum):
    AGENT: str
    LLM: str
    USER: str
    SYSTEM: str
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class DocMetaData(BaseModel):
    source: str
    is_chunk: bool
    id: str
    window_ids: list[str]
    def dict_bool_int(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    class Config:
        extra: Incomplete

class Document(BaseModel):
    content: str
    metadata: DocMetaData
    @staticmethod
    def hash_id(doc: str) -> str: ...
    def id(self) -> str: ...
