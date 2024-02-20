import pytest

from langroid.cachedb.redis_cachedb import RedisCache, RedisCacheConfig


@pytest.fixture
def fake_redis_cache():
    config = RedisCacheConfig(fake=False)
    cache = RedisCache(config=config)
    return cache


@pytest.mark.unit
def test_fake_store_and_retrieve(fake_redis_cache):
    key = "test_key"
    data = {"info": "something"}
    fake_redis_cache.store(key, data)
    result = fake_redis_cache.retrieve(key)
    assert result == data


@pytest.fixture
def real_redis_cache():
    config = RedisCacheConfig(
        fake=False,
    )
    cache = RedisCache(config=config)
    return cache


@pytest.mark.integration
def test_real_store_and_retrieve(real_redis_cache):
    key = "test_key"
    data = {"info": "something"}
    real_redis_cache.store(key, data)
    result = real_redis_cache.retrieve(key)
    assert result == data


@pytest.mark.integration
def test_key_deletion(real_redis_cache):
    keys = [f"_test_key_{i}" for i in range(10)]
    datas = [{f"info{i}": f"something{i}"} for i in range(10)]
    for key, data in zip(keys, datas):
        real_redis_cache.store(key, data)
    assert real_redis_cache.retrieve(keys[0]) is not None
    result = real_redis_cache.delete_keys(keys)
    assert result is None
    assert real_redis_cache.retrieve(keys[0]) is None

    for key, data in zip(keys, datas):
        real_redis_cache.store(key, data)
    assert real_redis_cache.retrieve(keys[0]) is not None
    result = real_redis_cache.delete_keys_pattern("_test_key_*")
    assert result is None
    assert real_redis_cache.retrieve(keys[0]) is None
