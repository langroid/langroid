import abc
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseSettings

class CacheDBConfig(BaseSettings): ...

class CacheDB(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def store(self, key: str, value: Any) -> None: ...
    @abstractmethod
    def retrieve(self, key: str) -> dict[str, Any] | str | None: ...
    @abstractmethod
    def delete_keys(self, keys: list[str]) -> None: ...
    @abstractmethod
    def delete_keys_pattern(self, pattern: str) -> None: ...
