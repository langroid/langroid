import difflib
import random
import re
from functools import cache
from itertools import islice
from typing import Any, Iterable, List

import nltk
from faker import Faker

Faker.seed(23)
random.seed(43)


# Ensures the NLTK resource is available
@cache
def download_nltk_resource(resource: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)


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
    download_nltk_resource("gutenberg")

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


def split_paragraphs(text: str) -> List[str]:
    """
    Split the input text into paragraphs using "\n\n" as the delimiter.

    Args:
        text (str): The input text.

    Returns:
        list: A list of paragraphs.
    """
    # Split based on a newline, followed by spaces/tabs, then another newline.
    paras = re.split(r"\n[ \t]*\n", text)
    return [para.strip() for para in paras if para.strip()]


def number_sentences(s: str) -> str:
    """
    Number the sentences in a given text, preserving paragraph structure.

    Args:
        s (str): The input text.

    Returns:
        str: The text with sentences numbered in the style (1), (2), etc.

    Example:
        >>> number_sentences("Hello world! How are you? Have a good day.")
        '(1) Hello world! (2) How are you? (3) Have a good day.'
    """
    numbered_text = []
    count = 1

    paragraphs = split_paragraphs(s)
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        for i, sentence in enumerate(sentences):
            sentence = f"({count}) {sentence}"
            count += 1
            sentences[i] = sentence
        numbered_paragraph = " ".join(sentences)
        numbered_text.append(numbered_paragraph)

    return "  \n\n  ".join(numbered_text)


def parse_number_range_list(specs: str) -> List[int]:
    """
    Parse a specs string like "3,5,7-10" into a list of integers.

    Args:
        specs (str): A string containing sentence numbers and/or ranges
                     (e.g., "3,5,7-10").

    Returns:
        List[int]: List of sentence numbers.

    Example:
        >>> parse_number_range_list("3,5,7-10")
        [3, 5, 7, 8, 9, 10]
    """
    spec_indices = set()  # type: ignore
    for part in specs.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            spec_indices.update(range(start, end + 1))
        else:
            spec_indices.add(int(part))

    return sorted(list(spec_indices))


def clean_whitespace(text: str) -> str:
    """Remove extra whitespace from the input text, while preserving
    paragraph structure.
    """
    paragraphs = split_paragraphs(text)
    cleaned_paragraphs = [" ".join(p.split()) for p in paragraphs if p]
    return "\n\n".join(cleaned_paragraphs)  # Join the cleaned paragraphs.


def extract_numbered_sentences(s: str, specs: str) -> str:
    """
    Extract specified sentences from a numbered text, preserving paragraph structure.

    Args:
        s (str): The input text containing numbered sentences.
        specs (str): A string containing sentence numbers and/or ranges
                     (e.g., "3,5,7-10").

    Returns:
        str: Extracted sentences, keeping original paragraph structures.

    Example:
        >>> text = "(1) Hello world! (2) How are you? (3) Have a good day."
        >>> extract_sentences(text, "1,3")
        'Hello world! Have a good day.'
    """
    # Use the helper function to get the list of indices from specs
    if specs.strip() == "":
        return ""
    spec_indices = parse_number_range_list(specs)

    # Regular expression to identify numbered sentences
    sentence_pattern = re.compile(r"\((\d+)\) ([^()]+[.!?])")

    # Split the text into paragraphs while preserving their boundaries
    paragraphs = split_paragraphs(s)

    extracted_paragraphs = []

    for paragraph in paragraphs:
        sentences_with_numbers = sentence_pattern.findall(paragraph)

        # Extract the desired sentences from this paragraph
        extracted_sentences = [
            sentence
            for num, sentence in sentences_with_numbers
            if int(num) in spec_indices
        ]

        # If we extracted any sentences from this paragraph,
        # join them and append to results
        if extracted_sentences:
            extracted_paragraphs.append(" ".join(extracted_sentences))

    return "\n\n".join(extracted_paragraphs)
