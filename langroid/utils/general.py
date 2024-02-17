from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

T = TypeVar("T")


def from_optional(x: Optional[T], default: T) -> T:
    """If `x` is None, returns `default`, else `x`."""
    if x is None:
        return default

    return x


def const(value: T) -> Callable[[Any], T]:
    """Returns a constant function which always returns `value`."""

    def fun(_: Any) -> T:
        return value

    return fun


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def add_grouped(
    groups: List[Tuple[T1, List[T2]]],
    key: T1,
    value: T2,
    pos: Optional[int] = None,
    group_pos: Optional[int] = None,
) -> None:
    """
    Incrementally constructs lists grouped by a key value. The value
    is added to the group in position pos if the key matches; else a
    new group is created. If pos is unspecified, adds the group at the
    end, else inserts it at position pos. The value is added at
    position group_pos (default appended to the end).
    """
    if len(groups) == 0:
        groups.append((key, [value]))
    else:
        idx = from_optional(pos, len(groups) - 1)
        idx_add = from_optional(pos, len(groups))

        group_key = groups[idx][0]
        prev_group_key = groups[idx - 1][0] if idx > 0 else None

        if key == group_key:
            group = groups[idx][1]
            idx_add = from_optional(group_pos, len(group))

            group.insert(idx_add, value)
        elif key == prev_group_key:
            group = groups[idx - 1][1]
            idx_add = from_optional(group_pos, len(group))

            group.insert(idx_add, value)
        else:
            groups.insert(idx_add, (key, [value]))


def noop(*args: List[Any], **kwargs: Dict[str, Any]) -> None:
    """Does nothing."""
    pass
