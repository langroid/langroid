import json
import logging
import os
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Dict, List, TypeVar

import fakeredis
import redis
from dotenv import load_dotenv
from pydantic import BaseModel

from langroid.cachedb.base import CacheDB

T = TypeVar("T", bound="RedisCache")
logger = logging.getLogger(__name__)


class RedisCacheConfig(BaseModel):
    """Configuration model for RedisCache."""

    fake: bool = False


class RedisCache(CacheDB):
    """Redis implementation of the CacheDB."""

    def __init__(self, config: RedisCacheConfig):
        """
        Initialize a RedisCache with the given config.

        Args:
            config (RedisCacheConfig): The configuration to use.
        """
        self.config = config
        load_dotenv()

        if self.config.fake:
            self.pool = fakeredis.FakeStrictRedis()  # type: ignore
        else:
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_host = os.getenv("REDIS_HOST")
            redis_port = os.getenv("REDIS_PORT")
            if None in [redis_password, redis_host, redis_port]:
                logger.warning(
                    """REDIS_PASSWORD, REDIS_HOST, REDIS_PORT not set in .env file,
                    using fake redis client"""
                )
                self.pool = fakeredis.FakeStrictRedis()  # type: ignore
            else:
                self.pool = redis.ConnectionPool(  # type: ignore
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                )

    @contextmanager  # type: ignore
    def redis_client(self) -> AbstractContextManager[T]:  # type: ignore
        """Cleanly open and close a redis client, avoids max clients exceeded error"""
        if isinstance(self.pool, fakeredis.FakeStrictRedis):
            yield self.pool
        else:
            client: T = redis.Redis(connection_pool=self.pool)
            try:
                yield client
            finally:
                client.close()

    def clear(self) -> None:
        """Clear keys from current db."""
        with self.redis_client() as client:  # type: ignore
            client.flushdb()

    def clear_all(self) -> None:
        """Clear all keys from all dbs."""
        with self.redis_client() as client:  # type: ignore
            client.flushall()

    def store(self, key: str, value: Any) -> None:
        """
        Store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (Any): The value to store.
        """
        with self.redis_client() as client:  # type: ignore
            client.set(key, json.dumps(value))

    def retrieve(self, key: str) -> Dict[str, Any] | str | None:
        """
        Retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        with self.redis_client() as client:  # type: ignore
            value = client.get(key)
            return json.loads(value) if value else None

    def delete_keys(self, keys: List[str]) -> None:
        """
        Delete the keys from the cache.

        Args:
            keys (List[str]): The keys to delete.
        """
        with self.redis_client() as client:  # type: ignore
            client.delete(*keys)
