from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction
from llmagent.embedding_models.base import EmbeddingModel
import dotenv
import os

class OpenAIEmbeddings(EmbeddingModel):
    @classmethod
    def embedding_fn(cls) -> EmbeddingFunction:
        dotenv.load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002",
        )
    @classmethod
    @property
    def embedding_dims(cls) -> int:
        return 1536

class SentenceTransformerEmbeddings(EmbeddingModel):
    @classmethod
    def embedding_fn(cls) -> EmbeddingFunction:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )

    @classmethod
    @property
    def embedding_dims(cls) -> int:
        return 384


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
