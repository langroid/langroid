import json
import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import momento
from dotenv import load_dotenv
from momento.responses import CacheGet
from pydantic import BaseModel

from langroid.cachedb.base import CacheDB

logger = logging.getLogger(__name__)


class MomentoCacheConfig(BaseModel):
    """Configuration model for RedisCache."""

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

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
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
