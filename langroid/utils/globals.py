from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="GlobalState")


class GlobalState(BaseModel):
    """A base Pydantic model for global states."""

    _instance: Optional["GlobalState"] = None

    @classmethod
    def get_instance(cls: Type["GlobalState"]) -> "GlobalState":
        """
        Get the global instance of the specific subclass.

        Returns:
            The global instance of the subclass.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_values(cls: Type[T], **kwargs: Dict[str, Any]) -> None:
        """
        Set values on the global instance of the specific subclass.

        Args:
            **kwargs: The fields and their values to set.
        """
        instance = cls.get_instance()
        for key, value in kwargs.items():
            setattr(instance, key, value)

    @classmethod
    def get_value(cls: Type[T], name: str) -> Any:
        """
        Retrieve the value of a specific field from the global instance.

        Args:
            name (str): The name of the field to retrieve.

        Returns:
            str: The value of the specified field.
        """
        instance = cls.get_instance()
        return getattr(instance, name)
