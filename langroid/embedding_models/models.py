import atexit
import os
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from langroid.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from langroid.exceptions import LangroidImportError
from langroid.mytypes import Embeddings
from langroid.parsing.utils import batched


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "openai"
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    api_base: Optional[str] = None
    organization: str = ""
    dims: int = 1536
    context_length: int = 8192


class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "sentence-transformer"
    model_name: str = "BAAI/bge-large-en-v1.5"
    context_length: int = 512
    data_parallel: bool = False
    # Select device (e.g. "cuda", "cpu") when data parallel is disabled
    device: Optional[str] = None
    # Select devices when data parallel is enabled
    devices: Optional[list[str]] = None


class FastEmbedEmbeddingsConfig(EmbeddingModelsConfig):
    """Config for qdrant/fastembed embeddings,
    see here: https://github.com/qdrant/fastembed
    """

    model_type: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 256
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    parallel: Optional[int] = None
    additional_kwargs: Dict[str, Any] = {}


class EmbeddingFunctionCallable:
    """
    A callable class designed to generate embeddings for a list of texts using
    the OpenAI API, with automatic retries on failure.

    Attributes:
        model (OpenAIEmbeddings): An instance of OpenAIEmbeddings that provides
                                configuration and utilities for generating embeddings.

    Methods:
        __call__(input: List[str]) -> Embeddings: Generate embeddings for
                                a list of input texts.
    """

    def __init__(self, model: "OpenAIEmbeddings", batch_size: int = 512):
        """
        Initialize the EmbeddingFunctionCallable with a specific model.

        Args:
            model (OpenAIEmbeddings): An instance of OpenAIEmbeddings to use for
            generating embeddings.
            batch_size (int): Batch size
        """
        self.model = model
        self.batch_size = batch_size

    def __call__(self, input: List[str]) -> Embeddings:
        """
        Generate embeddings for a given list of input texts using the OpenAI API,
        with retries on failure.

        This method:
        - Truncates each text in the input list to the model's maximum context length.
        - Processes the texts in batches to generate embeddings efficiently.
        - Automatically retries the embedding generation process with exponential
        backoff in case of failures.

        Args:
            input (List[str]): A list of input texts to generate embeddings for.

        Returns:
            Embeddings: A list of embedding vectors corresponding to the input texts.
        """
        tokenized_texts = self.model.truncate_texts(input)
        embeds = []
        for batch in batched(tokenized_texts, self.batch_size):
            result = self.model.client.embeddings.create(
                input=batch, model=self.model.config.model_name
            )
            batch_embeds = [d.embedding for d in result.data]
            embeds.extend(batch_embeds)
        return embeds


class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig = OpenAIEmbeddingsConfig()):
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("OPENAI_API_KEY", "")
        self.config.organization = os.getenv("OPENAI_ORGANIZATION", "")
        if self.config.api_key == "":
            raise ValueError(
                """OPENAI_API_KEY env variable must be set to use 
                OpenAIEmbeddings. Please set the OPENAI_API_KEY value 
                in your .env file.
                """
            )
        self.client = OpenAI(base_url=self.config.api_base, api_key=self.config.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def truncate_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Truncate texts to the embedding model's context length.
        TODO: Maybe we should show warning, and consider doing T5 summarization?
        """
        return [
            self.tokenizer.encode(text, disallowed_special=())[
                : self.config.context_length
            ]
            for text in texts
        ]

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunctionCallable(self, self.config.batch_size)

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


STEC = SentenceTransformerEmbeddingsConfig


class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, config: STEC = STEC()):
        # this is an "extra" optional dependency, so we import it here
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                """
                To use sentence_transformers embeddings, 
                you must install langroid with the [hf-embeddings] extra, e.g.:
                pip install "langroid[hf-embeddings]"
                """
            )

        super().__init__()
        self.config = config

        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        if self.config.data_parallel:
            self.pool = self.model.start_multi_process_pool(
                self.config.devices  # type: ignore
            )
            atexit.register(
                lambda: SentenceTransformer.stop_multi_process_pool(self.pool)
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.config.context_length = self.tokenizer.model_max_length

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        def fn(texts: List[str]) -> Embeddings:
            if self.config.data_parallel:
                embeds: Embeddings = self.model.encode_multi_process(
                    texts,
                    self.pool,
                    batch_size=self.config.batch_size,
                ).tolist()
            else:
                embeds = []
                for batch in batched(texts, self.config.batch_size):
                    batch_embeds = self.model.encode(
                        batch, convert_to_numpy=True
                    ).tolist()  # type: ignore
                    embeds.extend(batch_embeds)

            return embeds

        return fn

    @property
    def embedding_dims(self) -> int:
        dims = self.model.get_sentence_embedding_dimension()
        if dims is None:
            raise ValueError(
                f"Could not get embedding dimension for model {self.config.model_name}"
            )
        return dims  # type: ignore


class FastEmbedEmbeddings(EmbeddingModel):
    def __init__(self, config: FastEmbedEmbeddingsConfig = FastEmbedEmbeddingsConfig()):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise LangroidImportError("fastembed", extra="fastembed")

        super().__init__()
        self.config = config
        self._batch_size = config.batch_size
        self._parallel = config.parallel

        self._model = TextEmbedding(
            model_name=self.config.model_name,
            cache_dir=self.config.cache_dir,
            threads=self.config.threads,
            **self.config.additional_kwargs,
        )

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        def fn(texts: List[str]) -> Embeddings:
            embeddings = self._model.embed(
                texts, batch_size=self._batch_size, parallel=self._parallel
            )

            return [embedding.tolist() for embedding in embeddings]

        return fn

    @cached_property
    def embedding_dims(self) -> int:
        embed_func = self.embedding_fn()
        return len(embed_func(["text"])[0])


def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings  # type: ignore
    elif embedding_fn_type == "fastembed":
        return FastEmbedEmbeddings  # type: ignore
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings  # type: ignore
