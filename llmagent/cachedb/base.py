from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class CacheDB(ABC):
    """Abstract base class for a cache database."""

    @abstractmethod
    def store(self, key: str, value: Dict[str, Any]) -> None:
        """
        Abstract method to store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (dict): The value to store.
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Abstract method to retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        pass
