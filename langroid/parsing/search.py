"""
Utils to search for close matches in a list of strings.
"""

from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from rank_bm25 import BM25Okapi

from langroid.mytypes import Document


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
