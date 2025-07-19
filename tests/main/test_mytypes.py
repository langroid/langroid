import pytest

from langroid.mytypes import DocMetaData, Entity


@pytest.mark.parametrize("s", ["user", "User", "USER", "uSer", None])
def test_equality(s: str | None):
    if s is None:
        assert Entity.USER != s
        assert not Entity.USER == s
    else:
        assert Entity.USER == s
        assert not Entity.USER != s


def test_docmetadata_id_conversion():
    """Test that DocMetaData accepts various types for id and converts them
    to string.
    """
    # Test with integer id
    doc1 = DocMetaData(id=123)
    assert doc1.id == "123"
    assert isinstance(doc1.id, str)

    # Test with string id
    doc2 = DocMetaData(id="456")
    assert doc2.id == "456"
    assert isinstance(doc2.id, str)

    # Test with UUID-like string
    doc3 = DocMetaData(id="550e8400-e29b-41d4-a716-446655440000")
    assert doc3.id == "550e8400-e29b-41d4-a716-446655440000"
    assert isinstance(doc3.id, str)

    # Test with float (edge case)
    doc4 = DocMetaData(id=3.14)
    assert doc4.id == "3.14"
    assert isinstance(doc4.id, str)

    # Test with None (should be handled by default factory)
    doc5 = DocMetaData()
    assert isinstance(doc5.id, str)
    assert len(doc5.id) > 0  # Should have generated UUID

    # Test with zero
    doc6 = DocMetaData(id=0)
    assert doc6.id == "0"
    assert isinstance(doc6.id, str)

    # Test with negative number
    doc7 = DocMetaData(id=-1)
    assert doc7.id == "-1"
    assert isinstance(doc7.id, str)
