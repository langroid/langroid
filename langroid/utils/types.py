import json
import logging
from typing import Any, Optional, Type, TypeVar, Union, get_args, get_origin

from langroid.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)
PrimitiveType = Union[int, float, bool, str]
T = TypeVar("T")


def is_instance_of(obj: Any, type_hint: Type[T] | Any) -> bool:
    """
    Check if an object is an instance of a type hint, e.g.
    to check whether x is of type `List[ToolMessage]` or type `int`
    """
    if type_hint == Any:
        return True

    if type_hint is type(obj):
        return True

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Union:
        return any(is_instance_of(obj, arg) for arg in args)

    if origin:  # e.g. List, Dict, Tuple, Set
        if isinstance(obj, origin):
            # check if all items in obj are of the required types
            if args:
                if isinstance(obj, (list, tuple, set)):
                    return all(is_instance_of(item, args[0]) for item in obj)
                if isinstance(obj, dict):
                    return all(
                        is_instance_of(k, args[0]) and is_instance_of(v, args[1])
                        for k, v in obj.items()
                    )
            return True
        else:
            return False

    return isinstance(obj, type_hint)


def to_string(msg: Any) -> str:
    """
    Best-effort conversion of arbitrary msg to str.
    Return empty string if conversion fails.
    """
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, BaseModel):
        return msg.json()
    # last resort: use json.dumps() or str() to make it a str
    try:
        return json.dumps(msg)
    except Exception:
        try:
            return str(msg)
        except Exception as e:
            logger.error(
                f"""
                Error converting msg to str: {e}", 
                """,
                exc_info=True,
            )
            return ""


def from_string(
    s: str,
    output_type: Type[PrimitiveType],
) -> Optional[PrimitiveType]:
    if output_type is int:
        try:
            return int(s)
        except ValueError:
            return None
    elif output_type is float:
        try:
            return float(s)
        except ValueError:
            return None
    elif output_type is bool:
        return s.lower() in ("true", "yes", "1")
    elif output_type is str:
        return s
    else:
        return None
