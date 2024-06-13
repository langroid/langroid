import time
from typing import TYPE_CHECKING, Dict, Optional, TypeAlias, TypeVar
from uuid import uuid4

from langroid.pydantic_v1 import BaseModel

if TYPE_CHECKING:
    from langroid.agent.base import Agent
    from langroid.agent.chat_agent import ChatAgent
    from langroid.agent.chat_document import ChatDocument

    # any derivative of BaseModel that has an id() method or an id attribute
    ObjWithId: TypeAlias = ChatDocument | ChatAgent | Agent
else:
    ObjWithId = BaseModel

# Define a type variable that can be any subclass of BaseModel
T = TypeVar("T", bound=BaseModel)


class ObjectRegistry:
    """A global registry to hold id -> object mappings."""

    registry: Dict[str, ObjWithId] = {}

    @classmethod
    def add(cls, obj: ObjWithId) -> str:
        """Adds an object to the registry, returning the object's ID."""
        object_id = obj.id() if callable(obj.id) else obj.id
        cls.registry[object_id] = obj
        return object_id

    @classmethod
    def get(cls, obj_id: str) -> Optional[ObjWithId]:
        """Retrieves an object by ID if it still exists."""
        return cls.registry.get(obj_id)

    @classmethod
    def register_object(cls, obj: ObjWithId) -> str:
        """Registers an object in the registry, returning the object's ID."""
        return cls.add(obj)

    @classmethod
    def remove(cls, obj_id: str) -> None:
        """Removes an object from the registry."""
        if obj_id in cls.registry:
            del cls.registry[obj_id]

    @classmethod
    def cleanup(cls) -> None:
        """Cleans up the registry by removing entries where the object is None."""
        to_remove = [key for key, value in cls.registry.items() if value is None]
        for key in to_remove:
            del cls.registry[key]

    @staticmethod
    def new_id() -> str:
        """Generates a new unique ID."""
        return str(uuid4())


def scheduled_cleanup(interval: int = 600) -> None:
    """Periodically cleans up the global registry every 'interval' seconds."""
    while True:
        ObjectRegistry.cleanup()
        time.sleep(interval)
