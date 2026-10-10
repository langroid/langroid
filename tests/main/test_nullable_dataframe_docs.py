import pandas as pd
import pytest

from langroid.utils.pydantic_utils import dataframe_to_documents, first_non_null


@pytest.mark.parametrize(
    "dtype, value",
    [("Int64", 7), ("boolean", True), ("string", "tag"), ("float64", 1.5)],
)
def test_nullable_dataframe_fields_become_optional_document_values(dtype, value):
    frame = pd.DataFrame(
        {
            "text": ["first document", "second document"],
            "tag": pd.Series([None, value], dtype=dtype),
            "score": pd.Series([None, value], dtype=dtype),
        }
    )
    documents = dataframe_to_documents(frame, content="text", metadata=["tag"])

    assert len(documents) == 2
    assert documents[0].content == "first document"
    assert documents[0].metadata.tag is None
    assert documents[0].score is None
    assert documents[1].metadata.tag == value
    assert documents[1].score == value


def test_all_missing_columns_and_container_cells_remain_supported():
    frame = pd.DataFrame(
        {
            "content": ["first", "second"],
            "tag": pd.Series([pd.NA, pd.NA], dtype="string"),
            "values": [[1, 2], [3]],
        }
    )
    documents = dataframe_to_documents(frame, metadata=["tag"])
    assert [document.metadata.tag for document in documents] == [None, None]
    assert [document.values for document in documents] == [[1, 2], [3]]


def test_first_non_null_skips_scalar_missing_values_without_testing_container_truth():
    values = [1, 2]
    assert (
        first_non_null(pd.Series([None, pd.NA, float("nan"), pd.NaT, values])) is values
    )


@pytest.mark.parametrize(
    "dtype, value",
    [
        ("Int64", 7),
        ("boolean", True),
        ("string", "tag"),
        ("float64", 1.5),
        ("object", [1, 2]),
    ],
)
def test_model_from_all_missing_columns_accepts_values_from_later_batches(dtype, value):
    first_batch = pd.DataFrame(
        {
            "content": ["first"],
            "tag": pd.Series([None], dtype=dtype),
            "score": pd.Series([None], dtype=dtype),
        }
    )
    first_docs = dataframe_to_documents(first_batch, metadata=["tag"])
    assert first_docs[0].metadata.tag is None
    assert first_docs[0].score is None

    later_batch = pd.DataFrame(
        {
            "content": ["second"],
            "tag": pd.Series([value], dtype=dtype),
            "score": pd.Series([value], dtype=dtype),
        }
    )
    later_docs = dataframe_to_documents(
        later_batch, metadata=["tag"], doc_cls=type(first_docs[0])
    )
    assert later_docs[0].metadata.tag == value
    assert later_docs[0].score == value
