import pytest

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.search import (
    find_closest_matches_with_bm25,
    find_fuzzy_matches_in_docs,
    get_context,
    preprocess_text,
)


@pytest.fixture
def original_docs():
    return [
        Document(content="This is a sample blah document.", metadata=DocMetaData(id=1)),
        Document(content="Another example of document.", metadata=DocMetaData(id=2)),
        Document(content="Yet a another document sample.", metadata=DocMetaData(id=3)),
    ]


@pytest.fixture
def sample_docs():
    return [
        Document(content="This is a sample document.", metadata=DocMetaData(id=1)),
        Document(content="Another example of document.", metadata=DocMetaData(id=2)),
        Document(content="Yet another document sample.", metadata=DocMetaData(id=3)),
    ]


@pytest.mark.parametrize(
    "query, k, expected_length",
    [("sample", 1, 1), ("document", 2, 2), ("should not be found", 1, 0)],
)
def test_return_correct_number_of_matches(sample_docs, query, k, expected_length):
    results = find_fuzzy_matches_in_docs(query, sample_docs, k)
    assert len(results) == expected_length


@pytest.mark.parametrize(
    "words_before, words_after, expected",
    [
        (1, 1, "is a sample document"),
        (2, 2, "This is a sample document."),
        (None, None, "This is a sample document."),
    ],
)
def test_find_match_with_surrounding_words(
    sample_docs, words_before, words_after, expected
):
    query = "sample"
    k = 1
    results = find_fuzzy_matches_in_docs(
        query, sample_docs, k, words_before, words_after
    )
    assert expected in results[0].content


def test_empty_docs():
    docs = []
    docs_clean = []
    query = "test"
    result = find_closest_matches_with_bm25(docs, docs_clean, query)
    assert result == []


def test_matching_docs(sample_docs, original_docs):
    query = "test"
    result = find_closest_matches_with_bm25(original_docs, sample_docs, query, k=2)
    # As we are not mocking BM25Okapi, we can't predict exact scores.
    # We'll just assert that results are returned and are from our document list.
    assert len(result) == 2 and all(doc in original_docs for doc, score in result)


def test_preprocess_lowercase():
    result = preprocess_text("HELLO WORLD")
    assert result == "hello world"


def test_preprocess_remove_punctuation():
    result = preprocess_text("Hello, world!")
    assert result == "hello world"


# This test may vary depending on the actual stopwords list in nltk
def test_preprocess_remove_stopwords():
    result = preprocess_text("The world is a beautiful place.")
    assert "the" not in result
    assert "is" not in result
    assert "a " not in result


# This test assumes a default behavior of WordNetLemmatizer.
# It might need adjustments if lemmatization behavior changes in future nltk versions.
def test_preprocess_lemmatization():
    result = preprocess_text("running")
    assert "run" in result


def test_preprocess_combined():
    result = preprocess_text("The sun is shining, and birds are singing!")
    assert "the" not in result
    assert "is" not in result
    assert "and" not in result
    assert "sun" in result
    assert "bird" in result  # Assuming lemmatization converts "birds" to "bird"
    assert "sing" in result  # Assuming lemmatization converts "singing" to "sing"


@pytest.mark.parametrize(
    "query, text, before, after, expected, not_expected",
    [
        ("sample", "This is a sample document.", 1, 1, "a,document", "this"),
        (
            "UAS",
            "Develop a customizable Unmanned Aerial System (UAS) suite that will",
            0,
            3,
            "suite,that",
            "aerial,system",
        ),
    ],
)
def test_get_context(query, text, before, after, expected, not_expected):
    result = get_context(query, text, before, after)
    expected = expected.split(",")
    not_expected = not_expected.split(",")
    assert all(word in result for word in expected)
    assert all(word not in result for word in not_expected)
