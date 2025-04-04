import pytest

from langroid.utils.output.citations import invalid_markdown_citations


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("No citations here", []),
        ("Valid citation [^1] only", []),
        ("Invalid [^abc] citation", ["abc"]),
        ("Multiple [^x] [^y] [^z]", ["x", "y", "z"]),
        ("Mixed [^1] [^abc] [^2] [^xyz]", ["abc", "xyz"]),
        ("Duplicate [^x] [^x] [^y]", ["x", "y"]),
        ("[^abc123] [^123abc]", ["123abc", "abc123"]),  # Updated order to match sorting
        ("Ignore [^ ] empty and [^] blank", []),
    ],
)
def test_invalid_markdown_citations(input_str: str, expected: list[str]) -> None:
    """Test extraction of non-numeric markdown citations."""
    assert invalid_markdown_citations(input_str) == expected
