import json
import logging
import os
from typing import Any, Dict, Optional

import fakeredis
import redis
from dotenv import load_dotenv
from pydantic import BaseModel

from langroid.cachedb.base import CacheDB

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
            self.client = fakeredis.FakeStrictRedis()  # type: ignore
        else:
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_host = os.getenv("REDIS_HOST")
            redis_port = os.getenv("REDIS_PORT")
            if None in [redis_password, redis_host, redis_port]:
                logger.warning(
                    """REDIS_PASSWORD, REDIS_HOST, REDIS_PORT not set in .env file,
                    using fake redis client"""
                )
                self.client = fakeredis.FakeStrictRedis()  # type: ignore
            else:
                self.client = redis.Redis(  # type: ignore
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                )

    def clear(self) -> None:
        """Clear keys from current db."""
        self.client.flushdb()

    def clear_all(self) -> None:
        """Clear all keys from all dbs."""
        self.client.flushall()

    def store(self, key: str, value: Any) -> None:
        """
        Store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (Any): The value to store.
        """
        self.client.set(key, json.dumps(value))

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        value = self.client.get(key)
        return json.loads(value) if value else None
