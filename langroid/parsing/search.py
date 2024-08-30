"""
Utils to search for close matches in (a list of) strings.
Useful for retrieval of docs/chunks relevant to a query, in the context of
Retrieval-Augmented Generation (RAG), and SQLChat (e.g., to pull relevant parts of a
large schema).
See tests for examples: tests/main/test_string_search.py
"""

import difflib
from typing import List, Tuple

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from rank_bm25 import BM25Okapi
from thefuzz import fuzz, process

from langroid.mytypes import Document

from .utils import download_nltk_resource


def find_fuzzy_matches_in_docs(
    query: str,
    docs: List[Document],
    docs_clean: List[Document],
    k: int,
    words_before: int | None = None,
    words_after: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Find approximate matches of the query in the docs and return surrounding
    characters.

    Args:
        query (str): The search string.
        docs (List[Document]): List of Document objects to search through.
        docs_clean (List[Document]): List of Document objects with cleaned content.
        k (int): Number of best matches to return.
        words_before (int|None): Number of words to include before each match.
            Default None => return max
        words_after (int|None): Number of words to include after each match.
            Default None => return max

    Returns:
        List[Tuple[Document,float]]: List of (Document, score) tuples.
    """
    if len(docs) == 0:
        return []
    best_matches = process.extract(
        query,
        [d.content for d in docs_clean],
        limit=k,
        scorer=fuzz.partial_ratio,
    )

    real_matches = [(m, score) for m, score in best_matches if score > 50]
    # find the original docs that corresponding to the matches
    orig_doc_matches = []
    for i, (m, s) in enumerate(real_matches):
        for j, doc_clean in enumerate(docs_clean):
            if m in doc_clean.content:
                orig_doc_matches.append((docs[j], s))
                break
    if words_after is None and words_before is None:
        return orig_doc_matches
    if len(orig_doc_matches) == 0:
        return []
    if set(orig_doc_matches[0][0].__fields__) != {"content", "metadata"}:
        # If there are fields beyond just content and metadata,
        # we do NOT want to create new document objects with content fields
        # based on words_before and words_after, since we don't know how to
        # set those other fields.
        return orig_doc_matches

    contextual_matches = []
    for match, score in orig_doc_matches:
        choice_text = match.content
        contexts = []
        while choice_text != "":
            context, start_pos, end_pos = get_context(
                query, choice_text, words_before, words_after
            )
            if context == "" or end_pos == 0:
                break
            contexts.append(context)
            words = choice_text.split()
            end_pos = min(end_pos, len(words))
            choice_text = " ".join(words[end_pos:])
        if len(contexts) > 0:
            contextual_matches.append(
                (
                    Document(
                        content=" ... ".join(contexts),
                        metadata=match.metadata,
                    ),
                    score,
                )
            )

    return contextual_matches


def preprocess_text(text: str) -> str:
    """
    Preprocesses the given text by:
    1. Lowercasing all words.
    2. Tokenizing (splitting the text into words).
    3. Removing punctuation.
    4. Removing stopwords.
    5. Lemmatizing words.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed text.
    """
    # Ensure the NLTK resources are available
    for resource in ["punkt", "wordnet", "stopwords"]:
        download_nltk_resource(resource)

    # Lowercase the text
    text = text.lower()

    # Tokenize the text and remove punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Join the words back into a string
    text = " ".join(tokens)

    return text


def find_closest_matches_with_bm25(
    docs: List[Document],
    docs_clean: List[Document],
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Finds the k closest approximate matches using the BM25 algorithm.

    Args:
        docs (List[Document]): List of Documents to search through.
        docs_clean (List[Document]): List of cleaned Documents
        query (str): The search query.
        k (int, optional): Number of matches to retrieve. Defaults to 5.

    Returns:
        List[Tuple[Document,float]]: List of (Document, score) tuples.
    """
    if len(docs) == 0:
        return []
    texts = [doc.content for doc in docs_clean]
    query = preprocess_text(query)

    text_words = [text.split() for text in texts]

    bm25 = BM25Okapi(text_words)
    query_words = query.split()
    doc_scores = bm25.get_scores(query_words)

    # Get indices of top k scores
    top_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:k]

    # return the original docs, based on the scores from cleaned docs
    return [(docs[i], doc_scores[i]) for i in top_indices]


def get_context(
    query: str,
    text: str,
    words_before: int | None = 100,
    words_after: int | None = 100,
) -> Tuple[str, int, int]:
    """
    Returns a portion of text containing the best approximate match of the query,
    including b words before and a words after the match.

    Args:
    query (str): The string to search for.
    text (str): The body of text in which to search.
    b (int): The number of words before the query to return.
    a (int): The number of words after the query to return.

    Returns:
    str: A string containing b words before, the match, and a words after
        the best approximate match position of the query in the text. If no
        match is found, returns empty string.
    int: The start position of the match in the text.
    int: The end position of the match in the text.

    Example:
    >>> get_context("apple", "The quick brown fox jumps over the apple.", 3, 2)
    # 'fox jumps over the apple.'
    """
    if words_after is None and words_before is None:
        # return entire text since we're not asked to return a bounded context
        return text, 0, 0

    # make sure there is a good enough match to the query
    if fuzz.partial_ratio(query, text) < 40:
        return "", 0, 0

    sequence_matcher = difflib.SequenceMatcher(None, text, query)
    match = sequence_matcher.find_longest_match(0, len(text), 0, len(query))

    if match.size == 0:
        return "", 0, 0

    segments = text.split()
    n_segs = len(segments)

    start_segment_pos = len(text[: match.a].split())

    words_before = words_before or n_segs
    words_after = words_after or n_segs
    start_pos = max(0, start_segment_pos - words_before)
    end_pos = min(len(segments), start_segment_pos + words_after + len(query.split()))

    return " ".join(segments[start_pos:end_pos]), start_pos, end_pos


def eliminate_near_duplicates(passages: List[str], threshold: float = 0.8) -> List[str]:
    """
    Eliminate near duplicate text passages from a given list using MinHash and LSH.
    TODO: this has not been tested and the datasketch lib is not a dependency.
    Args:
        passages (List[str]): A list of text passages.
        threshold (float, optional): Jaccard similarity threshold to consider two
                                     passages as near-duplicates. Default is 0.8.

    Returns:
        List[str]: A list of passages after eliminating near duplicates.

    Example:
        passages = ["Hello world", "Hello, world!", "Hi there", "Hello world!"]
        print(eliminate_near_duplicates(passages))
        # ['Hello world', 'Hi there']
    """

    from datasketch import MinHash, MinHashLSH

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Create MinHash objects for each passage and insert to LSH
    minhashes = {}
    for idx, passage in enumerate(passages):
        m = MinHash(num_perm=128)
        for word in passage.split():
            m.update(word.encode("utf-8"))
        lsh.insert(idx, m)
        minhashes[idx] = m

    unique_idxs = set()
    for idx in minhashes.keys():
        # Query for similar passages (including itself)
        result = lsh.query(minhashes[idx])

        # If only the passage itself is returned, it's unique
        if len(result) == 1 and idx in result:
            unique_idxs.add(idx)

    return [passages[idx] for idx in unique_idxs]
