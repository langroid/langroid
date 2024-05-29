from abc import ABC
from typing import Any

from _typeshed import Incomplete
from pydantic import BaseModel

from langroid.language_models.base import LLMFunctionSpec as LLMFunctionSpec
from langroid.utils.pydantic_utils import (
    generate_simple_schema as generate_simple_schema,
)

class ToolMessage(ABC, BaseModel):
    request: str
    purpose: str
    result: str

    class Config:
        arbitrary_types_allowed: bool
        validate_all: bool
        validate_assignment: bool
        schema_extra: Incomplete

    @classmethod
    def instructions(cls) -> str: ...
    @classmethod
    def require_recipient(cls) -> type["ToolMessage"]: ...
    @classmethod
    def examples(cls) -> list["ToolMessage"]: ...
    @classmethod
    def usage_example(cls) -> str: ...
    def to_json(self) -> str: ...
    def json_example(self) -> str: ...
    def dict_example(self) -> dict[str, Any]: ...
    @classmethod
    def default_value(cls, f: str) -> Any: ...
    @classmethod
    def json_instructions(cls, tool: bool = False) -> str: ...
    @staticmethod
    def json_group_instructions() -> str: ...
    @classmethod
    def llm_function_schema(
        cls, request: bool = False, defaults: bool = True
    ) -> LLMFunctionSpec: ...
    @classmethod
    def simple_schema(cls) -> dict[str, Any]: ...
