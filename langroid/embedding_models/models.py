import atexit
import os
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

import requests
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from langroid.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from langroid.exceptions import LangroidImportError
from langroid.language_models.openai_gpt import LangDBParams
from langroid.mytypes import Embeddings
from langroid.parsing.utils import batched

AzureADTokenProvider = Callable[[], str]


class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "openai"
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: Optional[str] = None
    organization: str = ""
    dims: int = 1536
    context_length: int = 8192
    langdb_params: LangDBParams = LangDBParams()

    class Config:
        # enable auto-loading of env vars with OPENAI_ prefix, e.g.
        # api_base is set from OPENAI_API_BASE env var, in .env or system env
        env_prefix = "OPENAI_"


class AzureOpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "azure-openai"
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    deployment_name: Optional[str] = None
    # api_version defaulted to 2024-06-01 as per https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings?tabs=python-new
    # change this to required  supported version
    api_version: Optional[str] = "2024-06-01"
    # TODO: Add auth support for Azure OpenAI via AzureADTokenProvider
    azure_ad_token: Optional[str] = None
    azure_ad_token_provider: Optional[AzureADTokenProvider] = None
    dims: int = 1536
    context_length: int = 8192

    class Config:
        # enable auto-loading of env vars with AZURE_OPENAI_ prefix
        env_prefix = "AZURE_OPENAI_"


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


class LlamaCppServerEmbeddingsConfig(EmbeddingModelsConfig):
    api_base: str = ""
    context_length: int = 2048
    batch_size: int = 2048


class GeminiEmbeddingsConfig(EmbeddingModelsConfig):
    model_type: str = "gemini"
    model_name: str = "models/text-embedding-004"
    api_key: str = ""
    dims: int = 768
    batch_size: int = 512


class EmbeddingFunctionCallable:
    """
    A callable class designed to generate embeddings for a list of texts using
    the OpenAI or Azure OpenAI API, with automatic retries on failure.

    Attributes:
        embed_model (EmbeddingModel): An instance of EmbeddingModel that provides
               configuration and utilities for generating embeddings.

    Methods:
        __call__(input: List[str]) -> Embeddings: Generate embeddings for
                                a list of input texts.
    """

    def __init__(self, embed_model: EmbeddingModel, batch_size: int = 512):
        """
        Initialize the EmbeddingFunctionCallable with a specific model.

        Args:
            model ( OpenAIEmbeddings or AzureOpenAIEmbeddings): An instance of
                            OpenAIEmbeddings or AzureOpenAIEmbeddings to use for
                            generating embeddings.
            batch_size (int): Batch size
        """
        self.embed_model = embed_model
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
        embeds = []
        if isinstance(self.embed_model, (OpenAIEmbeddings, AzureOpenAIEmbeddings)):
            # Truncate texts to context length while preserving text format
            truncated_texts = self.embed_model.truncate_texts(input)

            # Process in batches
            for batch in batched(truncated_texts, self.batch_size):
                result = self.embed_model.client.embeddings.create(
                    input=batch, model=self.embed_model.config.model_name  # type: ignore
                )
                batch_embeds = [d.embedding for d in result.data]
                embeds.extend(batch_embeds)

        elif isinstance(self.embed_model, SentenceTransformerEmbeddings):
            if self.embed_model.config.data_parallel:
                embeds = self.embed_model.model.encode_multi_process(
                    input,
                    self.embed_model.pool,
                    batch_size=self.batch_size,
                ).tolist()
            else:
                for str_batch in batched(input, self.batch_size):
                    batch_embeds = self.embed_model.model.encode(
                        str_batch, convert_to_numpy=True
                    ).tolist()  # type: ignore
                    embeds.extend(batch_embeds)

        elif isinstance(self.embed_model, FastEmbedEmbeddings):
            embeddings = self.embed_model.model.embed(
                input, batch_size=self.batch_size, parallel=self.embed_model.parallel
            )

            embeds = [embedding.tolist() for embedding in embeddings]
        elif isinstance(self.embed_model, LlamaCppServerEmbeddings):
            for input_string in input:
                tokenized_text = self.embed_model.tokenize_string(input_string)
                for token_batch in batched(tokenized_text, self.batch_size):
                    gen_embedding = self.embed_model.generate_embedding(
                        self.embed_model.detokenize_string(list(token_batch))
                    )
                    embeds.append(gen_embedding)
        elif isinstance(self.embed_model, GeminiEmbeddings):
            embeds = self.embed_model.generate_embeddings(input)
        return embeds


class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig = OpenAIEmbeddingsConfig()):
        super().__init__()
        self.config = config
        load_dotenv()

        # Check if using LangDB
        self.is_langdb = self.config.model_name.startswith("langdb/")

        if self.is_langdb:
            self.config.model_name = self.config.model_name.replace("langdb/", "")
            self.config.api_base = self.config.langdb_params.base_url
            project_id = self.config.langdb_params.project_id
            if project_id:
                self.config.api_base += "/" + project_id + "/v1"
            self.config.api_key = self.config.langdb_params.api_key

        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY", "")

        self.config.organization = os.getenv("OPENAI_ORGANIZATION", "")

        if self.config.api_key == "":
            if self.is_langdb:
                raise ValueError(
                    """
                    LANGDB_API_KEY must be set in .env or your environment 
                    to use OpenAIEmbeddings via LangDB.
                    """
                )
            else:
                raise ValueError(
                    """
                    OPENAI_API_KEY must be set in .env or your environment 
                    to use OpenAIEmbeddings.
                    """
                )

        self.client = OpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
            organization=self.config.organization,
        )
        model_for_tokenizer = self.config.model_name
        if model_for_tokenizer.startswith("openai/"):
            self.config.model_name = model_for_tokenizer.replace("openai/", "")
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def truncate_texts(self, texts: List[str]) -> List[str] | List[List[int]]:
        """
        Truncate texts to the embedding model's context length.
        TODO: Maybe we should show warning, and consider doing T5 summarization?
        """
        truncated_tokens = [
            self.tokenizer.encode(text, disallowed_special=())[
                : self.config.context_length
            ]
            for text in texts
        ]

        if self.is_langdb:
            # LangDB embedding endpt only works with strings, not tokens
            return [self.tokenizer.decode(tokens) for tokens in truncated_tokens]
        return truncated_tokens

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunctionCallable(self, self.config.batch_size)

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


class AzureOpenAIEmbeddings(EmbeddingModel):
    """
    Azure OpenAI embeddings model implementation.
    """

    def __init__(
        self, config: AzureOpenAIEmbeddingsConfig = AzureOpenAIEmbeddingsConfig()
    ):
        """
        Initializes Azure OpenAI embeddings model.

        Args:
            config: Configuration for Azure OpenAI embeddings model.
        Raises:
            ValueError: If required Azure config values are not set.
        """
        super().__init__()
        self.config = config
        load_dotenv()

        if self.config.api_key == "":
            raise ValueError(
                """AZURE_OPENAI_API_KEY env variable must be set to use 
            AzureOpenAIEmbeddings. Please set the AZURE_OPENAI_API_KEY value 
            in your .env file."""
            )

        if self.config.api_base == "":
            raise ValueError(
                """AZURE_OPENAI_API_BASE env variable must be set to use 
            AzureOpenAIEmbeddings. Please set the AZURE_OPENAI_API_BASE value 
            in your .env file."""
            )
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.api_base,
            azure_deployment=self.config.deployment_name,
        )
        self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)

    def truncate_texts(self, texts: List[str]) -> List[str] | List[List[int]]:
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
        """Get the embedding function for Azure OpenAI.

        Returns:
            Callable that generates embeddings for input texts.
        """
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
        return EmbeddingFunctionCallable(self, self.config.batch_size)

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
        self.batch_size = config.batch_size
        self.parallel = config.parallel

        self.model = TextEmbedding(
            model_name=self.config.model_name,
            cache_dir=self.config.cache_dir,
            threads=self.config.threads,
            **self.config.additional_kwargs,
        )

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunctionCallable(self, self.config.batch_size)

    @cached_property
    def embedding_dims(self) -> int:
        embed_func = self.embedding_fn()
        return len(embed_func(["text"])[0])


LCSEC = LlamaCppServerEmbeddingsConfig


class LlamaCppServerEmbeddings(EmbeddingModel):
    def __init__(self, config: LCSEC = LCSEC()):
        super().__init__()
        self.config = config

        if self.config.api_base == "":
            raise ValueError(
                """Api Base MUST be set for Llama Server Embeddings.
                """
            )

        self.tokenize_url = self.config.api_base + "/tokenize"
        self.detokenize_url = self.config.api_base + "/detokenize"
        self.embedding_url = self.config.api_base + "/embeddings"

    def tokenize_string(self, text: str) -> List[int]:
        data = {"content": text, "add_special": False, "with_pieces": False}
        response = requests.post(self.tokenize_url, json=data)

        if response.status_code == 200:
            tokens = response.json()["tokens"]
            if not (isinstance(tokens, list) and isinstance(tokens[0], (int, float))):
                # not all(isinstance(token, (int, float)) for token in tokens):
                raise ValueError(
                    """Tokenizer endpoint has not returned the correct format. 
                   Is the URL correct?
                """
                )
            return tokens
        else:
            raise requests.HTTPError(
                self.tokenize_url,
                response.status_code,
                "Failed to connect to tokenization provider",
            )

    def detokenize_string(self, tokens: List[int]) -> str:
        data = {"tokens": tokens}
        response = requests.post(self.detokenize_url, json=data)

        if response.status_code == 200:
            text = response.json()["content"]
            if not isinstance(text, str):
                raise ValueError(
                    """Deokenizer endpoint has not returned the correct format. 
                   Is the URL correct?
                """
                )
            return text
        else:
            raise requests.HTTPError(
                self.detokenize_url,
                response.status_code,
                "Failed to connect to detokenization provider",
            )

    def truncate_string_to_context_size(self, text: str) -> str:
        tokens = self.tokenize_string(text)
        tokens = tokens[: self.config.context_length]
        return self.detokenize_string(tokens)

    def generate_embedding(self, text: str) -> List[int | float]:
        data = {"content": text}
        response = requests.post(self.embedding_url, json=data)

        if response.status_code == 200:
            embeddings = response.json()["embedding"]
            if not (
                isinstance(embeddings, list) and isinstance(embeddings[0], (int, float))
            ):
                raise ValueError(
                    """Embedding endpoint has not returned the correct format. 
                   Is the URL correct?
                """
                )
            return embeddings
        else:
            raise requests.HTTPError(
                self.embedding_url,
                response.status_code,
                "Failed to connect to embedding provider",
            )

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunctionCallable(self, self.config.batch_size)

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


class GeminiEmbeddings(EmbeddingModel):
    def __init__(self, config: GeminiEmbeddingsConfig = GeminiEmbeddingsConfig()):
        try:
            from google import genai
        except ImportError as e:
            raise LangroidImportError(extra="google-genai", error=str(e))
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("GEMINI_API_KEY", "")

        if self.config.api_key == "":
            raise ValueError(
                """
                GEMINI_API_KEY env variable must be set to use GeminiEmbeddings.
                """
            )
        self.client = genai.Client(api_key=self.config.api_key)

    def embedding_fn(self) -> Callable[[List[str]], Embeddings]:
        return EmbeddingFunctionCallable(self, self.config.batch_size)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of input texts."""
        all_embeddings: List[List[float]] = []

        for batch in batched(texts, self.config.batch_size):
            result = self.client.models.embed_content(  # type: ignore[attr-defined]
                model=self.config.model_name,
                contents=batch,
            )

            if not hasattr(result, "embeddings") or not isinstance(
                result.embeddings, list
            ):
                raise ValueError(
                    "Unexpected format for embeddings: missing or incorrect type"
                )

            # Extract .values from ContentEmbedding objects
            all_embeddings.extend([emb.values for emb in result.embeddings])

        return all_embeddings

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


def embedding_model(embedding_fn_type: str = "openai") -> EmbeddingModel:
    """
    Args:
        embedding_fn_type: Type of embedding model to use. Options are:
         - "openai",
         - "azure-openai",
         - "sentencetransformer", or
         - "fastembed".
            (others may be added in the future)
    Returns:
        EmbeddingModel: The corresponding embedding model class.
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings  # type: ignore
    elif embedding_fn_type == "azure-openai":
        return AzureOpenAIEmbeddings  # type: ignore
    elif embedding_fn_type == "fastembed":
        return FastEmbedEmbeddings  # type: ignore
    elif embedding_fn_type == "llamacppserver":
        return LlamaCppServerEmbeddings  # type: ignore
    elif embedding_fn_type == "gemini":
        return GeminiEmbeddings  # type: ignore
    else:  # default sentence transformer
        return SentenceTransformerEmbeddings  # type: ignore
