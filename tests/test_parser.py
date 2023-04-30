from llmagent.parsing.parser import ParsingConfig, Parser
from llmagent.mytypes import Document


def test_parser():
    cfg = ParsingConfig(
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

    docs = [Document(content=text, metadata={"id": i}) for i, text in enumerate(texts)]
    split_docs = parser.split(docs)
    assert len(split_docs) == 3
