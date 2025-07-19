from typing import Any, Dict, Optional, Type, TypeVar, cast

from pydantic import BaseModel
from pydantic.fields import ModelPrivateAttr
from pydantic_core import PydanticUndefined

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
        # Get the actual value from ModelPrivateAttr when accessing on class
        instance_attr = getattr(cls, "_instance", None)
        actual_instance: Optional["GlobalState"]
        if isinstance(instance_attr, ModelPrivateAttr):
            default_value = instance_attr.default
            if default_value is PydanticUndefined:
                actual_instance = None
            else:
                actual_instance = cast(Optional["GlobalState"], default_value)
        else:
            actual_instance = instance_attr

        if actual_instance is None:
            new_instance = cls()
            cls._instance = new_instance
            return new_instance
        return actual_instance  # type: ignore

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
