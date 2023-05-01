import pytest
from llmagent.parsing.parser import ParsingConfig, Parser
from llmagent.mytypes import Document
from llmagent.parsing.para_sentence_split import create_chunks
from typing import List


def test_parser():
    cfg = ParsingConfig(
        splitter="para_sentence",
        chunk_size=1000,
        chunk_overlap=2,
        separators=["."],
        token_encoding_model="text-davinci-003",
    )

    parser = Parser(cfg)
    texts = """
    This is a sentence. This is another sentence. This is a third sentence.
    """.strip().split(
        "."
    )
    texts = [text.strip() for text in texts if text.strip() != ""]

    docs = [Document(content=text, metadata={"id": i}) for i, text in enumerate(texts)]
    split_docs = parser.split(docs)
    assert len(split_docs) == 3


def length_fn(text):
    return len(text.split())


@pytest.mark.parametrize(
    "text, chunk_size, expected",
    [
        (
            "A B C D E F.\nG H I J K L.\n\nM N O P Q R.\nS T U V W X.",
            6,
            ["A B C D E F.", "G H I J K L.", "M N O P Q R.", "S T U V W X."],
        ),
        (
            "A B C D E F.\nG H I J K L.\n\nM N O P Q R.\nS T U V W X.",
            20,
            ["A B C D E F.\nG H I J K L.", "M N O P Q R.\nS T U V W X."],
        ),
    ],
)
def test_create_chunks(text: str, chunk_size: int, expected: List[str]):
    chunks = create_chunks(text, chunk_size, length_fn)
    assert chunks == expected
