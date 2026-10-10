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


def test_missing_value_normalization_skipped_when_nothing_is_missing(monkeypatch):
    """Rows with no missing value must not pay for the NA normalization.

    `dataframe_to_documents` calls `from_df_row` once per row, so normalizing
    unconditionally rebuilds two Series per row, which dominated ingestion of
    a large frame (0.85s -> 3.73s on 20k x 4 rows).

    Counting `Series.astype` rather than timing the two paths: a wall-clock
    comparison is the obvious test here, but it reads as a ratio between the
    clean and missing-value paths, and those converge on a loaded worker or
    wherever Pydantic validation is a larger share of the runtime -- so it can
    fail with the optimization perfectly intact. The counter is exact. It
    wraps the real method and calls through, so behavior is unchanged.
    """
    calls = []
    real_astype = pd.Series.astype

    def counting_astype(self, *args, **kwargs):
        calls.append(args[0] if args else kwargs.get("dtype"))
        return real_astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "astype", counting_astype, raising=True)

    columns = {
        "content": [f"text {i}" for i in range(4)],
        "num": list(range(4)),
        "score": [float(i) for i in range(4)],
        "tag": [f"s{i}" for i in range(4)],
    }

    clean = pd.DataFrame(columns)
    docs = dataframe_to_documents(clean, content="content", metadata=[])
    assert [doc.content for doc in docs] == ["text 0", "text 1", "text 2", "text 3"]
    assert calls == [], f"normalization ran on rows with nothing missing: {calls}"

    with_missing = pd.DataFrame(dict(columns))
    with_missing.loc[with_missing.index % 2 == 0, "tag"] = None
    docs = dataframe_to_documents(with_missing, content="content", metadata=[])
    assert [doc.tag for doc in docs] == [None, "s1", None, "s3"]
    # One `astype(object)` per row that actually has a missing value.
    assert calls.count(object) == 2, calls
