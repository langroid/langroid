from langroid.mytypes import DocMetaData, Document
from langroid.parsing.code_parser import CodeParser, CodeParsingConfig

MAX_CHUNK_SIZE = 10


def test_code_parser():
    cfg = CodeParsingConfig(
        chunk_size=MAX_CHUNK_SIZE,
        extensions=["py", "sh"],
        token_encoding_model="text-embedding-3-small",
    )

    parser = CodeParser(cfg)

    codes = """
    py|
    from pydantic import BaseModel
    from typing import List
    
    class Item(BaseModel):
        name: str
        description: str
        price: float
        tags: List[str]
    +
    py|
    import requests
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    +
    sh|
    #!/bin/bash

    # Function to prompt for user confirmation
    confirm() {
      read -p "$1 (y/n): " choice
      case "$choice" in
        [Yy]* ) return 0;;
        [Nn]* ) return 1;;
        * ) echo "Please answer y (yes) or n (no)."; return 1;;
      esac
    }
    """.split(
        "+"
    )

    codes = [text.strip() for text in codes if text.strip() != ""]
    lang_codes = [text.split("|") for text in codes]

    docs = [
        Document(content=code, metadata=DocMetaData(language=lang))
        for lang, code in lang_codes
        if code.strip() != ""
    ]
    split_docs = parser.split(docs)
    toks = parser.num_tokens
    # verify all chunks are less than twice max chunk size
    assert max([toks(doc.content) for doc in split_docs]) <= 2 * MAX_CHUNK_SIZE
    joined_splits = "".join([doc.content for doc in split_docs])
    joined_docs = "".join([doc.content for doc in docs])
    assert joined_splits.strip() == joined_docs.strip()
