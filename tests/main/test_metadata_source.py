import pytest

from langroid.agent.special.doc_chat_agent import _append_metadata_source


@pytest.mark.parametrize(
    ("original", "new", "expected"),
    [
        ("source-a", "source-a", "source-a"),
        ("  source-a  ", " source-a ", "source-a"),
        ("", "source-a", "source-a"),
        ("source-a", "", "source-a"),
        ("", "", ""),
        ("source-a", "source-b", "source-a; source-b"),
        ("https://host/a;b", "https://host/a;b", "https://host/a;b"),
        (
            "https://host/a;b; source-c",
            "https://host/a;b",
            "https://host/a;b; source-c",
        ),
        ("source-a; source-b", "source-b", "source-a; source-b"),
    ],
)
def test_append_metadata_source(
    original: str,
    new: str,
    expected: str,
) -> None:
    assert _append_metadata_source(original, new) == expected
