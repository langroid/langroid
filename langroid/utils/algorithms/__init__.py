from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import graph
else:
    graph = LazyLoad("langroid.utils.algorithms.graph")

__all__ = ["graph"]
