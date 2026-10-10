import numpy as np
import pandas as pd
import pytest

from langroid.utils.pydantic_utils import (
    dataframe_to_document_model,
    dataframe_to_documents,
    first_non_null,
)


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

    rows = 50
    columns = {
        "content": [f"text {i}" for i in range(rows)],
        "num": list(range(rows)),
        "score": [float(i) for i in range(rows)],
        "tag": [f"s{i}" for i in range(rows)],
    }

    clean = pd.DataFrame(columns)
    documents = dataframe_to_documents(clean, content="content", metadata=[])
    assert [doc.content for doc in documents] == columns["content"]
    # The assertion is that the work is bounded by COLUMNS, not rows -- that
    # is the property, and it does not pin how the normalization is split
    # between the frame and individual rows. Rewriting per row would be ~50
    # calls here rather than at most 4.
    assert len(calls) <= len(columns), f"normalization scales with rows: {calls}"

    with_missing = pd.DataFrame(dict(columns))
    with_missing.loc[with_missing.index % 2 == 0, "tag"] = None
    calls.clear()
    documents = dataframe_to_documents(with_missing, content="content", metadata=[])
    assert [doc.tag for doc in documents[:4]] == [None, "s1", None, "s3"]
    # A frame that does have missing values is still normalized, so this
    # cannot pass by never normalizing at all.
    assert calls.count(object) > 0, calls


@pytest.mark.parametrize("dtype", ["Int64", "int64"])
@pytest.mark.parametrize("value", [2**53 + 1, 2**63 - 1])
def test_large_integers_are_not_degraded_by_normalization(dtype, value):
    """Integers past 2**53 must survive ingestion exactly.

    `dataframe_to_documents` normalizes to `object` before handing values to
    Pydantic, which yields Python ints. Reaching Pydantic as `numpy.int64`
    instead silently rounds 2**53 + 1 down to 2**53, and fails validation
    outright at 2**63 - 1 -- so skipping the conversion to save work is only
    safe while these still hold.
    """
    frame = pd.DataFrame(
        {"content": ["t"], "big": pd.Series([value], dtype=dtype)},
    )

    documents = dataframe_to_documents(frame, content="content", metadata=[])

    assert documents[0].big == value
    assert isinstance(documents[0].big, int)


@pytest.mark.parametrize(
    "sentinel",
    [pd.NA, pd.NaT, float("nan"), np.float32("nan"), np.float64("nan")],
)
def test_from_df_row_normalizes_every_pandas_sentinel(sentinel):
    """Direct `from_df_row` callers must get None for any missing sentinel.

    `langroid/vector_store/lancedb.py` converts every query result through
    this classmethod, so it cannot rely on `dataframe_to_documents` having
    normalized the frame first. An earlier scan here tested only `pd.NA` and
    Python floats, which let `NaT` and `numpy.float32` NaN through as-is.
    """
    model = dataframe_to_document_model(
        pd.DataFrame({"content": ["a"], "x": [None]}),
        content="content",
        metadata=[],
    )
    row = pd.Series({"content": "a", "x": sentinel})

    document = model.from_df_row(row, "content", [])

    assert document.x is None


def test_from_df_row_leaves_container_cells_alone():
    """The sentinel scan must not compare container cells elementwise."""
    model = dataframe_to_document_model(
        pd.DataFrame({"content": ["a"], "x": [[1, 2]]}),
        content="content",
        metadata=[],
    )

    document = model.from_df_row(
        pd.Series({"content": "a", "x": [1, 2]}), "content", []
    )

    assert document.x == [1, 2]
