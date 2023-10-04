import difflib
import random
from itertools import islice
from typing import Any, Iterable, List, Callable

import nltk
from faker import Faker

Faker.seed(23)
random.seed(43)


# Wraps a function and ensures that it is run at most once
def run_once(f : Callable[[], None]) -> Callable[[], None]:
    has_run = False

    def wrapped():
        nonlocal has_run

        if not has_run:
            f()
            has_run = True

    return wrapped


# Ensure NLTK resources are available
@run_once
def download_nltk_resources() -> None:
    resources = ["punkt", "wordnet", "stopwords", "gutenberg"]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)


def batched(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def generate_random_sentences(k: int) -> str:
    # Load the sample text
    download_nltk_resources()

    from nltk.corpus import gutenberg

    text = gutenberg.raw("austen-emma.txt")

    # Split the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Generate k random sentences
    random_sentences = random.choices(sentences, k=k)
    return " ".join(random_sentences)


def generate_random_text(num_sentences: int) -> str:
    fake = Faker()
    text = ""
    for _ in range(num_sentences):
        text += fake.sentence() + " "
    return text


def closest_string(query: str, string_list: List[str]) -> str:
    """Find the closest match to the query in a list of strings.

    This function is case-insensitive and ignores leading and trailing whitespace.
    If no match is found, it returns 'No match found'.

    Args:
        query (str): The string to match.
        string_list (List[str]): The list of strings to search.

    Returns:
        str: The closest match to the query from the list, or 'No match found'
             if no match is found.
    """
    # Create a dictionary where the keys are the standardized strings and
    # the values are the original strings.
    str_dict = {s.lower().strip(): s for s in string_list}

    # Standardize the query and find the closest match in the list of keys.
    closest_match = difflib.get_close_matches(
        query.lower().strip(), str_dict.keys(), n=1
    )

    # Retrieve the original string from the value in the dictionary.
    original_closest_match = (
        str_dict[closest_match[0]] if closest_match else "No match found"
    )

    return original_closest_match
