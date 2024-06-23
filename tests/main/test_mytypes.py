import pytest

from langroid.mytypes import Entity


@pytest.mark.parametrize("s", ["user", "User", "USER", "uSer", None])
def test_equality(s: str | None):
    if s is None:
        assert Entity.USER != s
        assert not Entity.USER == s
    else:
        assert Entity.USER == s
        assert not Entity.USER != s
