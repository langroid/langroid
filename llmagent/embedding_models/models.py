from dataclasses import dataclass, field
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction
from llmagent.embedding_models.base import EmbeddingModel, EmbeddingModelsConfig
from dotenv import load_dotenv
import os


@dataclass
class OpenAIEmbeddingsConfig(EmbeddingModelsConfig):
    model_name:str = "text-embedding-ada-002"
    api_key: str = "" # field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    dims: int = 1536

@dataclass
class SentenceTransformerEmbeddingsConfig(EmbeddingModelsConfig):
    model_name:str = "all-MiniLM-L6-v2"
    dims: int = 384

class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, config: OpenAIEmbeddingsConfig):
        super().__init__()
        self.config = config
        load_dotenv()
        self.config.api_key = os.getenv("OPENAI_API_KEY")


    def embedding_fn(self) -> EmbeddingFunction:
        load_dotenv()
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.config.api_key,
            model_name=self.config.model_name,
        )

    @property
    def embedding_dims(self) -> int:
        return self.config.dims

class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, config: SentenceTransformerEmbeddingsConfig):
        super().__init__()
        self.config = config

    def embedding_fn(self) -> EmbeddingFunction:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.model_name,
        )

    @property
    def embedding_dims(self) -> int:
        return self.config.dims


def embedding_model(
        embedding_fn_type:str="openai"
) -> EmbeddingModel:
    """
    Thin wrapper around chromadb.utils.embedding_functions.
    Args:
        embedding_fn_type: "openai" or "sentencetransformer" # others soon
    Returns:
        EmbeddingModel
    """
    if embedding_fn_type == "openai":
        return OpenAIEmbeddings
    else: # default sentence transformer
        return SentenceTransformerEmbeddings
