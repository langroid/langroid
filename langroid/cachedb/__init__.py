from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import base
    from . import momento_cachedb
    from . import redis_cachedb
else:
    base = LazyLoad("langroid.cachedb.base")
    momento_cachedb = LazyLoad("langroid.cachedb.momento_cachedb")
    redis_cachedb = LazyLoad("langroid.cachedb.redis_cachedb")

__all__ = [
    "base",
    "momento_cachedb",
    "redis_cachedb",
]
