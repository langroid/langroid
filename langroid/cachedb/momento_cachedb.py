import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, List

from langroid.cachedb.base import CacheDBConfig
from langroid.exceptions import LangroidImportError

try:
    import momento
    from momento.responses import CacheGet
except ImportError:
    raise LangroidImportError(package="momento", extra="momento")

from dotenv import load_dotenv

from langroid.cachedb.base import CacheDB

logger = logging.getLogger(__name__)


class MomentoCacheConfig(CacheDBConfig):
    """Configuration model for Momento Cache."""

    ttl: int = 60 * 60 * 24 * 7  # 1 week
    cachename: str = "langroid_momento_cache"


class MomentoCache(CacheDB):
    """Momento implementation of the CacheDB."""

    def __init__(self, config: MomentoCacheConfig):
        """
        Initialize a MomentoCache with the given config.

        Args:
            config (MomentoCacheConfig): The configuration to use.
        """
        self.config = config
        load_dotenv()

        momento_token = os.getenv("MOMENTO_AUTH_TOKEN")
        if momento_token is None:
            raise ValueError("""MOMENTO_AUTH_TOKEN not set in .env file""")
        else:
            self.client = momento.CacheClient(
                configuration=momento.Configurations.Laptop.v1(),
                credential_provider=momento.CredentialProvider.from_environment_variable(
                    "MOMENTO_AUTH_TOKEN"
                ),
                default_ttl=timedelta(seconds=self.config.ttl),
            )
            self.client.create_cache(self.config.cachename)

    def clear(self) -> None:
        """Clear keys from current db."""
        self.client.flush_cache(self.config.cachename)

    def store(self, key: str, value: Any) -> None:
        """
        Store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (Any): The value to store.
        """
        self.client.set(self.config.cachename, key, json.dumps(value))

    def retrieve(self, key: str) -> Dict[str, Any] | str | None:
        """
        Retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        value = self.client.get(self.config.cachename, key)
        if isinstance(value, CacheGet.Hit):
            return json.loads(value.value_string)  # type: ignore
        else:
            return None

    def delete_keys(self, keys: List[str]) -> None:
        """
        Delete the keys from the cache.

        Args:
            keys (List[str]): The keys to delete.
        """
        for key in keys:
            self.client.delete(self.config.cachename, key)

    def delete_keys_pattern(self, pattern: str) -> None:
        """
        Delete the keys from the cache with the given pattern.

        Args:
            prefix (str): The pattern to match.
        """
        raise NotImplementedError(
            """
            MomentoCache does not support delete_keys_pattern.
            Please use RedisCache instead.
            """
        )
