"""
Test of:
GitHub Repo URL -> content files -> chunk
"""

from langroid.parsing.code_parser import CodeParser, CodeParsingConfig
from langroid.parsing.repo_loader import RepoLoader

MAX_CHUNK_SIZE = 20


def test_repo_chunking():
    url = "https://github.com/eugeneyan/testing-ml"
    repo_loader = RepoLoader(url)
    _, docs = repo_loader.load(depth=2, lines=100)
    assert len(docs) > 0

    parse_cfg = CodeParsingConfig(
        chunk_size=MAX_CHUNK_SIZE,
        extensions=["py", "sh", "md", "txt"],  # include text, code
        token_encoding_model="text-embedding-3-small",
    )

    parser = CodeParser(parse_cfg)
    split_docs = parser.split(docs)[:3]

    assert len(split_docs) > 0
