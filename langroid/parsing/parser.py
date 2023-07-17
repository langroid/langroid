import logging
from enum import Enum
from functools import reduce
from typing import List

import tiktoken
from pydantic import BaseSettings

from langroid.mytypes import Document
from langroid.parsing.para_sentence_split import create_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Splitter(str, Enum):
    TOKENS = "tokens"
    PARA_SENTENCE = "para_sentence"
    SIMPLE = "simple"


class ParsingConfig(BaseSettings):
    splitter: str = Splitter.TOKENS
    chunk_size: int = 200  # aim for this many tokens per chunk
    max_chunks: int = 10_000
    # aim to have at least this many chars per chunk when truncating due to punctuation
    min_chunk_chars: int = 350
    discard_chunk_chars: int = 5  # discard chunks with fewer than this many chars
    n_similar_docs: int = 4
    separators: List[str] = ["\n\n", "\n", " ", ""]
    token_encoding_model: str = "text-embedding-ada-002"


class Parser:
    def __init__(self, config: ParsingConfig):
        self.config = config
        self.tokenizer = tiktoken.encoding_for_model(config.token_encoding_model)

    def num_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def split_simple(self, docs: List[Document]) -> List[Document]:
        if len(self.config.separators) == 0:
            raise ValueError("Must have at least one separator")
        return [
            Document(content=chunk.strip(), metadata=d.metadata)
            for d in docs
            for chunk in d.content.split(self.config.separators[0])
            if chunk.strip() != ""
        ]

    def split_para_sentence(self, docs: List[Document]) -> List[Document]:
        final_chunks = []
        chunks = docs
        while True:
            long_chunks = [
                p
                for p in chunks
                if self.num_tokens(p.content) > 1.3 * self.config.chunk_size
            ]
            if len(long_chunks) == 0:
                break
            short_chunks = [
                p
                for p in chunks
                if self.num_tokens(p.content) <= 1.3 * self.config.chunk_size
            ]
            final_chunks += short_chunks
            chunks = self._split_para_sentence_once(long_chunks)
            if len(chunks) == len(long_chunks):
                max_len = max([self.num_tokens(p.content) for p in long_chunks])
                logger.warning(
                    f"""
                    Unable to split {len(long_chunks)} long chunks
                    using chunk_size = {self.config.chunk_size}.
                    Max chunk size is {max_len} tokens.
                    """
                )
                break  # we won't be able to shorten them with current settings

        return final_chunks + chunks

    def _split_para_sentence_once(self, docs: List[Document]) -> List[Document]:
        chunked_docs = [
            [
                Document(content=chunk.strip(), metadata=d.metadata)
                for chunk in create_chunks(
                    d.content, self.config.chunk_size, self.num_tokens
                )
                if chunk.strip() != ""
            ]
            for d in docs
        ]
        return reduce(lambda x, y: x + y, chunked_docs)

    def split_chunk_tokens(self, docs: List[Document]) -> List[Document]:
        chunked_docs = [
            [
                Document(content=chunk.strip(), metadata=d.metadata)
                for chunk in self.chunk_tokens(d.content)
                if chunk.strip() != ""
            ]
            for d in docs
        ]
        return reduce(lambda x, y: x + y, chunked_docs)

    def chunk_tokens(
        self,
        text: str,
    ) -> List[str]:
        """
        Split a text into chunks of ~CHUNK_SIZE tokens,
        based on punctuation and newline boundaries.
        Adapted from
        https://github.com/openai/chatgpt-retrieval-plugin/blob/main/services/chunks.py

        Args:
            text: The text to split into chunks.

        Returns:
            A list of text chunks, each of which is a string of tokens
            roughly self.config.chunk_size tokens long.
        """
        # Return an empty list if the text is empty or whitespace
        if not text or text.isspace():
            return []

        # Tokenize the text
        tokens = self.tokenizer.encode(text, disallowed_special=())

        # Initialize an empty list of chunks
        chunks = []

        # Initialize a counter for the number of chunks
        num_chunks = 0

        # Loop until all tokens are consumed
        while tokens and num_chunks < self.config.max_chunks:
            # Take the first chunk_size tokens as a chunk
            chunk = tokens[: self.config.chunk_size]

            # Decode the chunk into text
            chunk_text = self.tokenizer.decode(chunk)

            # Skip the chunk if it is empty or whitespace
            if not chunk_text or chunk_text.isspace():
                # Remove the tokens corresponding to the chunk text
                # from remaining tokens
                tokens = tokens[len(chunk) :]
                # Continue to the next iteration of the loop
                continue

            # Find the last period or punctuation mark in the chunk
            last_punctuation = max(
                chunk_text.rfind("."),
                chunk_text.rfind("?"),
                chunk_text.rfind("!"),
                chunk_text.rfind("\n"),
            )

            # If there is a punctuation mark, and the last punctuation index is
            # after MIN_CHUNK_SIZE_CHARS
            if (
                last_punctuation != -1
                and last_punctuation > self.config.min_chunk_chars
            ):
                # Truncate the chunk text at the punctuation mark
                chunk_text = chunk_text[: last_punctuation + 1]

            # Remove any newline characters and strip any leading or
            # trailing whitespace
            chunk_text_to_append = chunk_text.replace("\n", " ").strip()

            if len(chunk_text_to_append) > self.config.discard_chunk_chars:
                # Append the chunk text to the list of chunks
                chunks.append(chunk_text_to_append)

            # Remove the tokens corresponding to the chunk text
            # from the remaining tokens
            tokens = tokens[
                len(self.tokenizer.encode(chunk_text, disallowed_special=())) :
            ]

            # Increment the number of chunks
            num_chunks += 1

        # Handle the remaining tokens
        if tokens:
            remaining_text = self.tokenizer.decode(tokens).replace("\n", " ").strip()
            if len(remaining_text) > self.config.discard_chunk_chars:
                chunks.append(remaining_text)

        return chunks

    def split(self, docs: List[Document]) -> List[Document]:
        if len(docs) == 0:
            return []
        if self.config.splitter == Splitter.PARA_SENTENCE:
            return self.split_para_sentence(docs)
        elif self.config.splitter == Splitter.TOKENS:
            return self.split_chunk_tokens(docs)
        elif self.config.splitter == Splitter.SIMPLE:
            return self.split_simple(docs)
        else:
            raise ValueError(f"Unknown splitter: {self.config.splitter}")
