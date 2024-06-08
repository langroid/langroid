from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langroid.pydantic_v1 import BaseSettings


class CacheDBConfig(BaseSettings):
    """Configuration model for CacheDB."""

    pass


class CacheDB(ABC):
    """Abstract base class for a cache database."""

    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """
        Abstract method to store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (Any): The value to store.
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Dict[str, Any] | str | None:
        """
        Abstract method to retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        pass

    @abstractmethod
    def delete_keys(self, keys: List[str]) -> None:
        """
        Delete the keys from the cache.

        Args:
            keys (List[str]): The keys to delete.
        """
        pass

    @abstractmethod
    def delete_keys_pattern(self, pattern: str) -> None:
        """
        Delete all keys with the given pattern

        Args:
            prefix (str): The pattern to match.
        """
        pass
