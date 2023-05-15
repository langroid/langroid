import redis
import json
from pydantic import BaseModel
import fakeredis
from llmagent.cachedb.base import CacheDB
from dotenv import load_dotenv
import os

class RedisCacheConfig(BaseModel):
    """Configuration model for RedisCache."""
    fake: bool = False
    hostname: str = "redis-11524.c251.east-us-mz.azure.cloud.redislabs.com"
    port: int = 11524


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
        redis_password = os.getenv("REDIS_PASSWORD")
        if self.config.fake:
            self.client = fakeredis.FakeStrictRedis()
        else:
            self.client = redis.Redis(
                host=self.config.hostname,
                port=self.config.port,
                password=redis_password,
            )

    def store(self, key: str, value: dict) -> None:
        """
        Store a value associated with a key.

        Args:
            key (str): The key under which to store the value.
            value (dict): The value to store.
        """
        self.client.set(key, json.dumps(value))

    def retrieve(self, key: str) -> dict:
        """
        Retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            dict: The value associated with the key.
        """
        value = self.client.get(key)
        return json.loads(value) if value else None
