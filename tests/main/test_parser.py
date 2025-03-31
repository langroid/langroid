import tempfile

import pytest

from langroid.mytypes import Document
from langroid.parsing.parser import Parser, ParsingConfig, Splitter
from langroid.parsing.utils import extract_content_from_path, generate_random_text

CHUNK_SIZE = 100


@pytest.mark.parametrize(
    "splitter, chunk_size, max_chunks, min_chunk_chars, discard_chunk_chars",
    [
        # (Splitter.TOKENS, 10, 100, 35, 2),
        (Splitter.PARA_SENTENCE, 10, 3000, 35, 2),
        (Splitter.SIMPLE, 10, 500 * 5, 35, 2),
    ],
)
def test_parser(
    splitter: Splitter,
    chunk_size: int,
    max_chunks: int,
    min_chunk_chars: int,
    discard_chunk_chars: int,
):
    cfg = ParsingConfig(
        splitter=splitter,
        n_neighbor_ids=2,
        chunk_size_variation=0.2,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        separators=["."],
        min_chunk_chars=min_chunk_chars,
        discard_chunk_chars=discard_chunk_chars,
        token_encoding_model="text-embedding-3-small",
    )

    parser = Parser(cfg)
    docs = [
        Document(content=generate_random_text(500), metadata={"id": i})
        for i in range(5)
    ]

    split_docs = parser.split(docs)

    chunk_size_upper_bound = (
        chunk_size * (1 + cfg.chunk_size_variation)
        if splitter == Splitter.MARKDOWN
        else chunk_size + 5
    )
    assert all(
        parser.num_tokens(d.content) <= chunk_size_upper_bound for d in split_docs
    )
    assert len(split_docs) <= max_chunks * len(docs)
    assert all(len(d.content) >= discard_chunk_chars for d in split_docs)
    assert all(d.metadata.is_chunk for d in split_docs)

    # test neighbor chunks
    doc = Document(content=generate_random_text(500), metadata={"id": 0})
    chunks = parser.split([doc])
    n = len(chunks)
    if n > 2 * cfg.n_neighbor_ids + 1:
        assert len(chunks[n // 2].metadata.window_ids) == 2 * cfg.n_neighbor_ids + 1


def length_fn(text):
    return len(text.split())  # num chars


@pytest.mark.parametrize(
    "chunk_size, max_chunks, min_chunk_chars, discard_chunk_chars",
    [
        (100, 10_000, 350, 5),
        (10, 100, 35, 2),
        (200, 1000, 300, 10),
    ],
)
def test_text_token_chunking(
    chunk_size: int, max_chunks: int, min_chunk_chars: int, discard_chunk_chars: int
):
    cfg = ParsingConfig(
        chunk_size=chunk_size,
        max_chunks=max_chunks,
        min_chunk_chars=min_chunk_chars,
        discard_chunk_chars=discard_chunk_chars,
        token_encoding_model="text-embedding-3-small",
    )

    parser = Parser(cfg)

    text = generate_random_text(60)
    chunks = parser.chunk_tokens(text)

    assert len(chunks) <= max_chunks
    assert all(len(c) >= discard_chunk_chars for c in chunks)
    assert all(parser.num_tokens(c) <= chunk_size + 5 for c in chunks)


def test_extract_content():
    parsing = ParsingConfig()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file1:
        file1.write("Hello world")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file2:
        file2.write("It was the best of times")

    # extract from single path
    content = extract_content_from_path(file1.name, parsing)
    assert "Hello" in content

    # extract from multiple paths
    contents = extract_content_from_path([file1.name, file2.name], parsing)
    assert "Hello" in contents[0]
    assert "best" in contents[1]

    # read bytes from file1
    with open(file1.name, "rb") as file1:
        bytes_content1 = file1.read()

    with open(file2.name, "rb") as file2:
        bytes_content2 = file2.read()

    content = extract_content_from_path(bytes_content1, parsing)
    assert "Hello" in content

    contents = extract_content_from_path([bytes_content1, bytes_content2], parsing)
    assert "Hello" in contents[0]
    assert "best" in contents[1]


def test_utf8():
    my_str = "abc﷽🤦🏻‍♂️🤦🏻‍♂️🤦🏻‍♂️"
    b = my_str.encode("utf-8")  # 57 bytes that represent 19 chars
    content = b[:50]  # choose to cut it off at 50 for this example

    def find_last_full_char(str_to_test):
        for i in range(len(str_to_test) - 1, 0, -1):
            if (str_to_test[i] & 0xC0) != 0x80:
                return i

    content = content[: find_last_full_char(content)]

    # test that this succeeds
    _ = content.decode("utf-8")


def test_chunk_tokens():
    """Tests if Parser.chunk_tokens preserves list structure and line formatting."""
    cfg = ParsingConfig(
        chunk_size=10,
        max_chunks=5,
        min_chunk_chars=5,
        discard_chunk_chars=2,
        token_encoding_model="text-embedding-3-small",
    )
    parser = Parser(cfg)

    # text with bullet list, redundant extra lines
    text = """fruits
- apple

- orange



vegetables
- tomato
- cucumber"""

    chunks = parser.chunk_tokens(text)
    reconstructed = "".join(chunks)

    original_lines = [line.strip() for line in text.split("\n") if line.strip()]
    result_lines = [line.strip() for line in reconstructed.split("\n") if line.strip()]

    assert original_lines == result_lines
    assert len(original_lines) == 6  # Verify all lines are present
    assert all(line.startswith("- ") for line in original_lines if "-" in line)
