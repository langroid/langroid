import re
from typing import Callable, List

from bs4 import BeautifulSoup


def remove_extra_whitespace(s: str) -> str:
    lines = s.split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(cleaned_lines)


def custom_sent_tokenize(text: str) -> List[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"\.\s|\.\n", text)
        if sentence.strip()
    ]
    # append a period if the sentence does not end with one
    return [s + "." if s[-1] != "." else s for s in sentences]


def create_chunks(
    text: str, chunk_size: int, length_fn: Callable[[str], int]
) -> List[str]:
    def _chunk_sentences(sentences: List[str], chunk_size: int) -> List[str]:
        chunks = []
        current_chunk: List[str] = []
        current_chunk_length = 0

        for sentence in sentences:
            sentence_length = length_fn(sentence)
            if current_chunk_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_chunk_length += sentence_length

        if current_chunk:
            new_chunk = " ".join(current_chunk).strip()
            if new_chunk:
                chunks.append(" ".join(current_chunk).strip())

        return chunks

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # First, try to split the document into paragraphs
    paragraphs = text.split("\n\n")

    # If paragraphs are too long, split them into sentences
    if any(length_fn(p) > chunk_size for p in paragraphs):
        sentences = custom_sent_tokenize(text)
        chunks = _chunk_sentences(sentences, chunk_size)
    else:
        chunks = paragraphs

    chunks = [chunk.strip() for chunk in chunks if chunk.strip() != ""]
    return chunks
