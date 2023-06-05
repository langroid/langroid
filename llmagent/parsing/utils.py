import random

import nltk
from faker import Faker

nltk.download("punkt")
nltk.download("gutenberg")

Faker.seed(23)
random.seed(43)


def generate_random_sentences(k: int) -> str:
    # Load the sample text
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
