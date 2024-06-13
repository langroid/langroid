import pandas as pd

from langroid.mytypes import DocMetaData, Document
from langroid.utils.configuration import Settings, set_global
from langroid.utils.pydantic_utils import dataframe_to_documents


def test_df_to_documents(test_settings: Settings):
    set_global(test_settings)

    data = {
        "id": ["A100", "B200", "C300", "D400", "E500"],
        "year": [1955, 1977, 1989, 2001, 2015],
        "author": [
            "Isaac Asimov",
            "J.K. Rowling",
            "George Orwell",
            "J.R.R. Tolkien",
            "H.G. Wells",
        ],
        "title": [
            "The Last Question",
            "Harry Potter",
            "1984",
            "The Lord of the Rings",
            "The Time Machine",
        ],
        "summary": [
            "A story exploring the concept of entropy and the end of the universe.",
            "The adventures of a young wizard and his friends at a magical school.",
            "A dystopian novel about a totalitarian regime and the concept of freedom.",
            "An epic fantasy tale of a quest to destroy a powerful ring.",
            "A science fiction novel about time travel and its consequences.",
        ],
    }

    df = pd.DataFrame(data)

    docs = dataframe_to_documents(df, content="summary", metadata=["id", "year"])
    assert len(docs) == 5
    assert docs[0].content == data["summary"][0]
    assert docs[0].metadata.id == data["id"][0]
    assert docs[0].metadata.year == data["year"][0]
    assert docs[0].author == data["author"][0]
    assert isinstance(docs[0], Document)
    assert isinstance(docs[0].metadata, DocMetaData)

    # Note: "id" cannot be used at top level within Document class
    # since `id` is also the name of a method in the Document class
    df = df.drop(columns=["id"], inplace=False)
    docs = dataframe_to_documents(df, content="junk", metadata=[])
    assert len(docs) == 5
    assert docs[0].content == ""  # since `junk` is not a column in the dataframe
    assert docs[0].year == data["year"][0]
    assert docs[0].author == data["author"][0]
    assert isinstance(docs[0], Document)
    assert isinstance(docs[0].metadata, DocMetaData)
