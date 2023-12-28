import os
from typing import Callable, List

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from langroid.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from langroid.language_models.utils import retry_with_exponential_backoff
from langroid.mytypes import Embeddings
from langroid.parsing.utils import batched


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "openai"
    model_name: str = "text-embedding-ada-002"
    api_key: str = ""
    organization: str = ""
    dims: int = 1536
    context_length: int = 8192


class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "sentence-transformer"
    model_name: str = "BAAI/bge-large-en-v1.5"
    context_length: int = 512


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
        self.client = OpenAI(api_key=self.config.api_key)
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
        @retry_with_exponential_backoff
        def fn(texts: List[str]) -> Embeddings:
            tokenized_texts = self.truncate_texts(texts)
            embeds = []
            for batch in batched(tokenized_texts, 500):
                result = self.client.embeddings.create(
                    input=batch, model=self.config.model_name
                )
                batch_embeds = [d.embedding for d in result.data]
                embeds.extend(batch_embeds)
            return embeds

        return fn

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
        self.model = SentenceTransformer(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.config.context_length = self.tokenizer.model_max_length

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        def fn(texts: List[str]) -> Embeddings:
            embeds = []
            for batch in batched(texts, 500):
                batch_embeds = self.model.encode(batch, convert_to_numpy=True).tolist()
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


def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings  # type: ignore
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings  # type: ignore
