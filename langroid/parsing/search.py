"""
Utils to search for close matches in a list of strings.
"""

from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from rank_bm25 import BM25Okapi
from thefuzz import fuzz, process

from langroid.mytypes import Document


def find_fuzzy_matches_in_docs(
    query: str, docs: List[Document], k: int, surrounding_words: int
) -> List[Document]:
    """
    Find approximate matches of the query in the docs and return surrounding
    characters.

    Args:
        query (str): The search string.
        docs (List[Document]): List of Document objects to search through.
        k (int): Number of best matches to return.
        surrounding_words (int): Number of words to
            include before and after each match.

    Returns:
        List[Document]: List of Documents containing the matches,
            including the given number of words around the match.
    """
    best_matches = process.extract(
        query,
        [d.content for d in docs],
        limit=k,
        scorer=fuzz.partial_token_sort_ratio,
    )

    results = []
    for match, _ in best_matches:
        words = match.split()
        for doc in docs:
            if match in doc.content:
                words_in_text = doc.content.split()
                first_word_idx = next(
                    (
                        i
                        for i, word in enumerate(words_in_text)
                        if word.startswith(words[0])
                    ),
                    -1,
                )
                if first_word_idx != -1:
                    start_idx = max(0, first_word_idx - surrounding_words)
                    end_idx = min(
                        len(words_in_text),
                        first_word_idx + len(words) + surrounding_words,
                    )
                    doc_match = Document(
                        content=" ".join(words_in_text[start_idx:end_idx]),
                        metadata=doc.metadata,
                    )
                    results.append(doc_match)
                break

    return results


# Ensure NLTK resources are available
def download_nltk_resources() -> None:
    resources = ["punkt", "wordnet", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


download_nltk_resources()


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
