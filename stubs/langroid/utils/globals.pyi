from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="GlobalState")

class GlobalState(BaseModel):
    @classmethod
    def get_instance(cls) -> GlobalState: ...
    @classmethod
    def set_values(cls, **kwargs: dict[str, Any]) -> None: ...
    @classmethod
    def get_value(cls, name: str) -> Any: ...
