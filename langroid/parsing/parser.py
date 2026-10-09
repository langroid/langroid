import logging
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import tiktoken
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from langroid.mytypes import Document
from langroid.parsing.md_parser import (
    MarkdownChunkConfig,
    chunk_markdown,
    count_words,
)
from langroid.parsing.para_sentence_split import create_chunks, remove_extra_whitespace
from langroid.utils.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Splitter(str, Enum):
    TOKENS = "tokens"
    PARA_SENTENCE = "para_sentence"
    SIMPLE = "simple"
    # "structure-aware" splitting with chunks enriched by header info
    MARKDOWN = "markdown"


class BaseParsingConfig(BaseSettings):
    """Base class for document parsing configurations."""

    library: str

    model_config = SettingsConfigDict(extra="ignore")  # Ignore unknown settings


class LLMPdfParserConfig(BaseSettings):
    """Configuration for LLM-based parsing."""

    model_name: str = "gemini/gemini-2.0-flash"  # Default model
    max_tokens: Optional[int] = None
    split_on_page: Optional[bool] = True
    requests_per_minute: Optional[int] = 5
    timeout: int = 60
    prompt: str = ""  # override with a domain-specific prompt
    system_prompt: str = ""  # override with a domain-specific system prompt


class MarkerConfig(BaseSettings):
    """Configuration for Markitdown-based parsing."""

    config_dict: Dict[str, Any] = {}


class PdfParsingConfig(BaseParsingConfig):
    library: Literal[
        "fitz",
        "pymupdf4llm",
        "pypdfium2",
        "docling",
        "pypdf",
        "unstructured",
        "pdf2image",
        "markitdown",
        "llm-pdf-parser",
        "marker",
    ] = "pypdfium2"
    llm_parser_config: Optional[LLMPdfParserConfig] = None
    marker_config: Optional[MarkerConfig] = None

    @model_validator(mode="before")
    @classmethod
    def enable_configs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure correct config is set based on library selection."""
        library = values.get("library")

        if library == "llm-pdf-parser":
            values.setdefault("llm_parser_config", LLMPdfParserConfig())
        else:
            values["llm_parser_config"] = None

        if library == "marker":
            values.setdefault("marker_config", MarkerConfig())
        else:
            values["marker_config"] = None

        return values


class DocxParsingConfig(BaseSettings):
    library: Literal["python-docx", "unstructured", "markitdown-docx"] = "unstructured"


class DocParsingConfig(BaseSettings):
    library: Literal["unstructured"] = "unstructured"


class MarkitdownPPTXParsingConfig(BaseSettings):
    library: Literal["markitdown"] = "markitdown"


class MarkitdownXLSXParsingConfig(BaseSettings):
    library: Literal["markitdown"] = "markitdown"


class MarkitdownXLSParsingConfig(BaseSettings):
    library: Literal["markitdown"] = "markitdown"


class ParsingConfig(BaseSettings):
    splitter: str = Splitter.MARKDOWN
    chunk_by_page: bool = False  # split by page?
    chunk_size: int = 200  # aim for this many tokens per chunk
    chunk_size_variation: float = 0.30  # max variation from chunk_size
    overlap: int = 50  # overlap between chunks
    max_chunks: int = 10_000
    url_connect_timeout: float = 10.0
    url_read_timeout: float = 30.0
    url_max_size: int = 10 * 1024 * 1024
    # Cap on the total number of bytes a ZIP-based document (DOCX, XLSX, PPTX)
    # may expand to. `url_max_size` bounds only the *compressed* bytes fetched
    # over HTTP, and local paths / raw bytes are not bounded at all, so without
    # this a small archive can expand into an arbitrary amount of parsing work.
    # Set to 0 to disable the check.
    zip_max_expanded_size: int = 100 * 1024 * 1024

    @field_validator("chunk_size", mode="before")
    @classmethod
    def convert_chunk_size_to_int(cls, v: Any) -> int:
        """Convert chunk_size to int, maintaining backward compatibility
        with Pydantic V1.
        """
        if isinstance(v, float):
            return int(v)
        return int(v)

    # offset to subtract from page numbers:
    # e.g. if physical page 12 is displayed as page 1, set page_number_offset = 11
    page_number_offset: int = 0
    # aim to have at least this many chars per chunk when truncating due to punctuation
    min_chunk_chars: int = 350
    discard_chunk_chars: int = 5  # discard chunks with fewer than this many chars
    n_similar_docs: Optional[int] = None  # deprecated
    n_neighbor_ids: int = 5  # window size to store around each chunk
    separators: List[str] = ["\n\n", "\n", " ", ""]
    token_encoding_model: str = "text-embedding-3-small"
    pdf: PdfParsingConfig = PdfParsingConfig()
    docx: DocxParsingConfig = DocxParsingConfig()
    doc: DocParsingConfig = DocParsingConfig()
    pptx: MarkitdownPPTXParsingConfig = MarkitdownPPTXParsingConfig()
    xls: MarkitdownXLSParsingConfig = MarkitdownXLSParsingConfig()
    xlsx: MarkitdownXLSXParsingConfig = MarkitdownXLSXParsingConfig()


def add_window_ids(chunks: List[Document], n_neighbor_ids: int) -> None:
    """Give each chunk a distinct id and the ids of its neighbors.

    Chunks may belong to multiple docs, but for each doc they appear
    consecutively, and all chunks of one doc share a `metadata.id` -- which
    is how they are grouped here, so a window never crosses a document.

    Already-windowed chunks are left alone. Grouping by a shared
    `metadata.id` is what makes this non-idempotent: after one call every
    chunk has its own id, so a second call would see each chunk as a
    single-chunk document and collapse every window to a singleton,
    silently discarding the neighbors from the first call. A chunk is taken
    to be already windowed when it is alone under its id and that id is in
    its own window -- the exact state this function leaves behind, and one
    a group of siblings sharing an inherited id cannot be in.

    The decision is per group, not per call, so mixing windowed chunks with
    fresh ones windows only the fresh ones.

    Args:
        chunks: chunks to window, modified in place.
        n_neighbor_ids: how many neighbors to keep on each side.
    """
    # discard empty chunks
    chunks = [c for c in chunks if c.content.strip() != ""]
    if len(chunks) == 0:
        return

    # group by the original metadata.id
    # (each distinct orig_id refers to a different document)
    orig_id_to_chunks: Dict[str, List[Document]] = {}
    for c in chunks:
        orig_id_to_chunks.setdefault(c.metadata.id, []).append(c)

    k = n_neighbor_ids
    for orig_id, group in orig_id_to_chunks.items():
        if len(group) == 1 and orig_id in group[0].metadata.window_ids:
            # already windowed by an earlier call; re-running would replace
            # its window with a singleton rather than refine it. is_chunk is
            # still set, since this function has always set it on everything
            # it was handed.
            group[0].metadata.is_chunk = True
            continue
        # The original metadata.id is ignored since it is the same for all
        # chunks of a doc and is useless. We want a distinct id for each.
        ids = [ObjectRegistry.new_id() for _ in group]
        n = len(ids)
        window_ids = [ids[max(0, i - k) : min(n, i + k + 1)] for i in range(n)]
        for i, c in enumerate(group):
            c.metadata.window_ids = window_ids[i]
            c.metadata.id = ids[i]
            c.metadata.is_chunk = True


class Parser:
    def __init__(self, config: ParsingConfig):
        self.config = config
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.token_encoding_model)
        except Exception:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

    def num_tokens(self, text: str) -> int:
        if self.config.splitter == Splitter.MARKDOWN:
            return count_words(text)  # simple count based on whitespace-split
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return len(tokens)

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def add_window_ids(self, chunks: List[Document]) -> None:
        """Chunks may belong to multiple docs, but for each doc,
        they appear consecutively. Add window_ids in metadata.

        Chunks this has already windowed are left as they are, so calling
        it twice does not replace their neighbors with a singleton. See
        the module-level `add_window_ids` for the details.
        """
        add_window_ids(chunks, self.config.n_neighbor_ids)

    def split_simple(self, docs: List[Document]) -> List[Document]:
        if len(self.config.separators) == 0:
            raise ValueError("Must have at least one separator")
        final_docs = []

        for d in docs:
            if d.content.strip() == "":
                continue
            chunks = remove_extra_whitespace(d.content).split(self.config.separators[0])
            # note we are ensuring we COPY the document metadata into each chunk,
            # which ensures all chunks of a given doc have same metadata
            # (and in particular same metadata.id, which is important later for
            # add_window_ids)
            chunk_docs = [
                Document(
                    content=c,
                    metadata=d.metadata.model_copy(
                        # window_ids cleared, not inherited: a chunk's
                        # window names its own siblings
                        update=dict(is_chunk=True, window_ids=[])
                    ),
                )
                for c in chunks
                if c.strip() != ""
            ]
            self.add_window_ids(chunk_docs)
            final_docs += chunk_docs
        return final_docs

    def split_para_sentence(self, docs: List[Document]) -> List[Document]:
        chunks = docs
        while True:
            un_splittables = 0
            split_chunks = []
            for c in chunks:
                if c.content.strip() == "":
                    continue
                if self.num_tokens(c.content) <= 1.3 * self.config.chunk_size:
                    # small chunk: no need to split
                    split_chunks.append(c)
                    continue
                splits = self._split_para_sentence_once([c])
                un_splittables += len(splits) == 1
                split_chunks += splits
            if len(split_chunks) == len(chunks):
                if un_splittables > 0:
                    max_len = max([self.num_tokens(p.content) for p in chunks])
                    logger.warning(
                        f"""
                        Unable to split {un_splittables} chunks
                        using chunk_size = {self.config.chunk_size}.
                        Max chunk size is {max_len} tokens.
                        """
                    )
                break  # we won't be able to shorten them with current settings
            chunks = split_chunks.copy()

        self.add_window_ids(chunks)
        return chunks

    def _split_para_sentence_once(self, docs: List[Document]) -> List[Document]:
        final_chunks = []
        for d in docs:
            if d.content.strip() == "":
                continue
            chunks = create_chunks(d.content, self.config.chunk_size, self.num_tokens)
            # note we are ensuring we COPY the document metadata into each chunk,
            # which ensures all chunks of a given doc have same metadata
            # (and in particular same metadata.id, which is important later for
            # add_window_ids)
            chunk_docs = [
                Document(
                    content=c,
                    metadata=d.metadata.model_copy(
                        # window_ids cleared, not inherited: a chunk's
                        # window names its own siblings
                        update=dict(is_chunk=True, window_ids=[])
                    ),
                )
                for c in chunks
                if c.strip() != ""
            ]
            final_chunks += chunk_docs

        return final_chunks

    def split_chunk_tokens(self, docs: List[Document]) -> List[Document]:
        final_docs = []
        for d in docs:
            if self.config.splitter == Splitter.MARKDOWN:
                chunks = chunk_markdown(
                    d.content,
                    MarkdownChunkConfig(
                        # apply rough adjustment factor to convert from tokens to words,
                        # which is what the markdown chunker uses
                        chunk_size=int(self.config.chunk_size * 0.75),
                        overlap_tokens=int(self.config.overlap * 0.75),
                        variation_percent=self.config.chunk_size_variation,
                        rollup=True,
                    ),
                )
            else:
                chunks = self.chunk_tokens(d.content)
            # note we are ensuring we COPY the document metadata into each chunk,
            # which ensures all chunks of a given doc have same metadata
            # (and in particular same metadata.id, which is important later for
            # add_window_ids)
            chunk_docs = [
                Document(
                    content=c,
                    metadata=d.metadata.model_copy(
                        # window_ids cleared, not inherited: a chunk's
                        # window names its own siblings
                        update=dict(is_chunk=True, window_ids=[])
                    ),
                )
                for c in chunks
                if c.strip() != ""
            ]
            self.add_window_ids(chunk_docs)
            final_docs += chunk_docs
        return final_docs

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
            punctuation_matches = [
                (m.start(), m.group())
                for m in re.finditer(r"(?:[.!?][\s\n]|\n)", chunk_text)
            ]

            last_punctuation = max([pos for pos, _ in punctuation_matches] + [-1])

            # If there is a punctuation mark, and the last punctuation index is
            # after MIN_CHUNK_SIZE_CHARS
            if (
                last_punctuation != -1
                and last_punctuation > self.config.min_chunk_chars
            ):
                # Truncate the chunk text at the punctuation mark
                chunk_text = chunk_text[: last_punctuation + 1]

            # Replace redundant (3 or more) newlines with 2 newlines to preser
            # paragraph separation!
            # But do NOT strip leading/trailing whitespace, to preserve formatting
            # (e.g. code blocks, or in case we want to stitch chunks back together)
            chunk_text_to_append = re.sub(r"\n{3,}", "\n\n", chunk_text)

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

        # There may be remaining tokens, but we discard them
        # since we have already reached the maximum number of chunks

        return chunks

    def split(self, docs: List[Document]) -> List[Document]:
        if len(docs) == 0:
            return []
        # create ids in metadata of docs if absent:
        # we need this to distinguish docs later in add_window_ids
        for d in docs:
            if d.metadata.id in [None, ""]:
                d.metadata.id = ObjectRegistry.new_id()
        # some docs are already splits, so don't split them further!
        chunked_docs = [d for d in docs if d.metadata.is_chunk]
        big_docs = [d for d in docs if not d.metadata.is_chunk]
        if len(big_docs) == 0:
            return chunked_docs
        match self.config.splitter:
            case Splitter.MARKDOWN | Splitter.TOKENS:
                big_doc_chunks = self.split_chunk_tokens(big_docs)
            case Splitter.PARA_SENTENCE:
                big_doc_chunks = self.split_para_sentence(big_docs)
            case Splitter.SIMPLE:
                big_doc_chunks = self.split_simple(big_docs)
            case _:
                raise ValueError(f"Unknown splitter: {self.config.splitter}")

        return chunked_docs + big_doc_chunks
