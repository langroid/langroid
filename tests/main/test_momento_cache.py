
import pytest

from langroid.cachedb.momento_cachedb import MomentoCache, MomentoCacheConfig


@pytest.fixture
def momento_cache():
    config = MomentoCacheConfig()
    cache = MomentoCache(config=config)
    return cache


@pytest.mark.integration
def test_real_store_and_retrieve(momento_cache):
    key = "test_key"
    data = {"info": "something"}
    momento_cache.store(key, data)
    result = momento_cache.retrieve(key)
    assert result == data
