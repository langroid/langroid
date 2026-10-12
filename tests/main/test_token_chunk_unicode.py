import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.parser import Parser, ParsingConfig, Splitter


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100, 200])
@pytest.mark.parametrize(
    "text",
    [
        "你好世界",
        "こんにちは世界",
        "🦛🦙🍎 hello",
        "مرحبا بالعالم",
        "café naïve e\u0301",
        "prefix " + "中🦛文 " * 120 + "suffix",
        "An ASCII document with spaces and punctuation.\nNext line.",
        "A literal replacement character: \ufffd and more text",
    ],
)
def test_token_chunks_preserve_unicode(text: str, chunk_size: int) -> None:
    """Splitting a UTF-8 character across BPE tokens must not corrupt it."""
    parser = Parser(
        ParsingConfig(
            splitter=Splitter.TOKENS,
            chunk_size=chunk_size,
            min_chunk_chars=10_000,
            discard_chunk_chars=0,
        )
    )
    assert "".join(parser.chunk_tokens(text)) == text

    document = Document(content=text, metadata=DocMetaData(source="unicode.txt"))
    chunks = parser.split([document])
    assert "".join(chunk.content for chunk in chunks) == text
    assert all(chunk.metadata.source == "unicode.txt" for chunk in chunks)
    assert all(chunk.metadata.is_chunk for chunk in chunks)


@pytest.mark.parametrize("max_chunks", [1, 2])
def test_token_chunk_limit_ends_at_complete_character(max_chunks: int) -> None:
    parser = Parser(
        ParsingConfig(
            splitter=Splitter.TOKENS,
            chunk_size=1,
            max_chunks=max_chunks,
            min_chunk_chars=10_000,
            discard_chunk_chars=0,
        )
    )
    assert parser.chunk_tokens("🦛🦛🦛") == ["🦛"] * max_chunks


@pytest.mark.parametrize("min_chunk_chars", [0, 10_000])
@pytest.mark.parametrize("chunk_size", [2, 7, 25])
def test_token_chunks_advance_by_original_tokens(
    min_chunk_chars: int, chunk_size: int
) -> None:
    """Re-encoding a decoded prefix can produce fewer tokens and repeat text."""
    text = (
        "ZZZaYabb0?XZcbY2aZ1!.?a2b2.1!..b1YZa!b0!01233cZX03a1232Z0Yc3.c.3Y3?"
        "02X32bZa32XZ! More text. More punctuation! Tail."
    )
    parser = Parser(
        ParsingConfig(
            splitter=Splitter.TOKENS,
            chunk_size=chunk_size,
            min_chunk_chars=min_chunk_chars,
            discard_chunk_chars=0,
        )
    )
    assert "".join(parser.chunk_tokens(text)) == text
