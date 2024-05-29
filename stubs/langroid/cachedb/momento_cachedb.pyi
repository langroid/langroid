from typing import Any

from _typeshed import Incomplete

from langroid.cachedb.base import CacheDB as CacheDB
from langroid.cachedb.base import CacheDBConfig as CacheDBConfig
from langroid.exceptions import LangroidImportError as LangroidImportError

logger: Incomplete

class MomentoCacheConfig(CacheDBConfig):
    ttl: int
    cachename: str

class MomentoCache(CacheDB):
    config: Incomplete
    client: Incomplete
    def __init__(self, config: MomentoCacheConfig) -> None: ...
    def clear(self) -> None: ...
    def store(self, key: str, value: Any) -> None: ...
    def retrieve(self, key: str) -> dict[str, Any] | str | None: ...
    def delete_keys(self, keys: list[str]) -> None: ...
    def delete_keys_pattern(self, pattern: str) -> None: ...
